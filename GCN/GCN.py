from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	from torch_geometric.nn import GCNConv
except ImportError as exc:
	raise ImportError(
		"torch_geometric is required. Install it with: pip install torch-geometric"
	) from exc


FEATURE_COLS = [
	"ret_1",
	"ret_5",
	"ret_10",
	"ret_20",
	"hl_spread",
	"co_ret",
	"log_vol",
	"turnover_rate",
	"volume_ratio",
	"pb",
	"ma5_gap",
	"ma10_gap",
	"ma20_gap",
	"volatility_5",
	"volatility_20",
]


class GCN(nn.Module):
	def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.0):
		super().__init__()
		self.conv1 = GCNConv(in_dim, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, 1)
		self.dropout = dropout

	def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
		x = self.conv1(x, edge_index)
		x = torch.relu(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.conv2(x, edge_index)
		return x


def parse_args() -> argparse.Namespace:
	repo_root = Path(__file__).resolve().parents[1]
	parser = argparse.ArgumentParser(
		description="Train a GCN using prebuilt graph dataset files."
	)
	parser.add_argument(
		"--graph-data-dir",
		type=Path,
		default=repo_root / "build_graph" / "graph_dataset",
		help="Directory containing metadata.pt/train.pt/val.pt/test.pt.",
	)
	parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
	parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate.")
	parser.add_argument(
		"--weight-decay",
		type=float,
		default=1e-4,
		help="L2 regularization for Adam.",
	)
	parser.add_argument(
		"--dropout",
		type=float,
		default=0.0,
		help="Dropout probability after first GCN layer.",
	)
	parser.add_argument(
		"--early-stopping-patience",
		type=int,
		default=20,
		help="Stop training if validation IC does not improve for this many epochs.",
	)
	parser.add_argument(
		"--topk-list",
		type=str,
		default="10,30,50",
		help="Comma-separated Top-k list for cross-sectional backtest metrics (IRR/Sharpe).",
	)
	parser.add_argument(
		"--device",
		type=str,
		choices=["cuda"],
		default="cuda",
		help="Training device. CUDA is required.",
	)
	parser.add_argument(
		"--save-model-path",
		type=Path,
		default=repo_root / "GCN" / "gcn_model.pt",
		help="Path to save trained model checkpoint.",
	)
	parser.add_argument(
		"--output-log",
		type=Path,
		default=repo_root / "GCN" / "gcn_train_log.txt",
		help="Path to save training log.",
	)
	return parser.parse_args()


def parse_topk_list(topk_list: str) -> list[int]:
	values: list[int] = []
	for token in topk_list.split(","):
		token = token.strip()
		if not token:
			continue
		k = int(token)
		if k <= 0:
			raise ValueError("All Top-k values must be positive integers.")
		values.append(k)

	if not values:
		raise ValueError("--topk-list must contain at least one positive integer.")

	return list(dict.fromkeys(values))


def resolve_device(device_name: str) -> torch.device:
	if device_name != "cuda":
		raise ValueError("This script requires --device cuda.")
	if not torch.cuda.is_available():
		raise RuntimeError("CUDA requested but not available.")
	torch.cuda.set_device(0)
	return torch.device("cuda:0")


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
	if len(a) <= 1:
		return float("nan")
	if not np.isfinite(a).all() or not np.isfinite(b).all():
		return float("nan")
	if np.std(a) == 0 or np.std(b) == 0:
		return float("nan")
	return float(np.corrcoef(a, b)[0, 1])


def load_graph_dataset(
	graph_data_dir: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
	if not graph_data_dir.exists():
		raise FileNotFoundError(f"Graph dataset directory not found: {graph_data_dir}")

	meta_path = graph_data_dir / "metadata.pt"
	train_path = graph_data_dir / "train.pt"
	val_path = graph_data_dir / "val.pt"
	test_path = graph_data_dir / "test.pt"

	for p in [meta_path, train_path, val_path, test_path]:
		if not p.exists():
			raise FileNotFoundError(f"Required dataset file not found: {p}")

	metadata = torch.load(meta_path, map_location="cpu")
	train_set = torch.load(train_path, map_location="cpu")
	val_set = torch.load(val_path, map_location="cpu")
	test_set = torch.load(test_path, map_location="cpu")

	for split_name, split_data in [
		("train", train_set),
		("val", val_set),
		("test", test_set),
	]:
		for key in ["x", "y", "mask", "trade_dates"]:
			if key not in split_data:
				raise KeyError(f"{split_name}.pt missing key: {key}")

	for key in ["edge_index", "node_codes", "feature_cols", "target_col"]:
		if key not in metadata:
			raise KeyError(f"metadata.pt missing key: {key}")

	return metadata, train_set, val_set, test_set


def train(
	model: nn.Module,
	train_set: dict[str, object],
	val_set: dict[str, object],
	edge_index: torch.Tensor,
	device: torch.device,
	epochs: int,
	lr: float,
 	weight_decay: float,
 	early_stopping_patience: int,
) -> tuple[list[str], dict[str, torch.Tensor], float, int]:
	model.to(device)
	edge_index = edge_index.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

	x_all = train_set["x"]
	y_all = train_set["y"]
	mask_all = train_set["mask"]

	if not isinstance(x_all, torch.Tensor) or not isinstance(y_all, torch.Tensor):
		raise TypeError("train.pt keys 'x' and 'y' must be torch.Tensor")
	if not isinstance(mask_all, torch.Tensor):
		raise TypeError("train.pt key 'mask' must be torch.Tensor")

	if x_all.ndim != 3:
		raise ValueError("train x tensor must have shape [T, N, F].")
	if y_all.ndim != 2 or mask_all.ndim != 2:
		raise ValueError("train y/mask tensors must have shape [T, N].")

	n_steps = x_all.shape[0]
	if n_steps == 0:
		raise ValueError("train split is empty. Cannot train model.")

	best_state_dict = copy.deepcopy(model.state_dict())
	best_val_ic = float("-inf")
	best_epoch = 0
	patience = 0

	val_y_true, val_y_pred, _ = predict_split(model, val_set, edge_index, device)
	if len(val_y_true) == 0:
		raise ValueError("validation split is empty. Cannot perform early stopping.")

	logs: list[str] = []
	for epoch in range(1, epochs + 1):
		model.train()
		epoch_loss_sum = 0.0
		epoch_point_cnt = 0

		for t in range(n_steps):
			x = x_all[t].to(device)
			y = y_all[t].to(device)
			mask = mask_all[t].to(device)

			pred = model(x, edge_index).squeeze(-1)
			if int(mask.sum().item()) == 0:
				continue

			loss = ((pred[mask] - y[mask]) ** 2).mean()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			n_obs = int(mask.sum().item())
			epoch_loss_sum += float(loss.item()) * n_obs
			epoch_point_cnt += n_obs

		avg_loss = epoch_loss_sum / max(epoch_point_cnt, 1)
		val_y_true, val_y_pred, _ = predict_split(model, val_set, edge_index, device)
		val_metrics = regression_metrics(val_y_true, val_y_pred)
		current_val_ic = val_metrics["ic"]

		if current_val_ic > best_val_ic:
			best_val_ic = current_val_ic
			best_state_dict = copy.deepcopy(model.state_dict())
			best_epoch = epoch
			patience = 0
		else:
			patience += 1

		pred_mean = float(np.mean(val_y_pred)) if len(val_y_pred) > 0 else float("nan")
		pred_std = float(np.std(val_y_pred)) if len(val_y_pred) > 1 else float("nan")
		pred_abs_max = float(np.max(np.abs(val_y_pred))) if len(val_y_pred) > 0 else float("nan")

		line = (
			f"epoch={epoch:03d} mse={avg_loss:.8f} | val_ic={current_val_ic:.8f} | "
			f"best_val_ic={best_val_ic:.8f} | best_epoch={best_epoch:03d} | "
			f"val_pred_mean={pred_mean:.8f} | val_pred_std={pred_std:.8f} | "
			f"val_pred_abs_max={pred_abs_max:.8f}"
		)
		logs.append(line)
		print(line)

		if patience >= early_stopping_patience:
			logs.append(f"early_stopping_triggered_at_epoch={epoch:03d}")
			print(f"early_stopping_triggered_at_epoch={epoch:03d}")
			break

	return logs, best_state_dict, best_val_ic, best_epoch


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
	residual = y_true - y_pred
	mse = float(np.mean(residual**2))
	sign_accuracy = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

	ic = safe_corr(y_pred, y_true)
	pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=np.float64)
	true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=np.float64)
	rankic = safe_corr(pred_rank, true_rank)

	strategy_ret = np.sign(y_pred) * y_true
	ret_std = float(np.std(strategy_ret, ddof=1)) if len(strategy_ret) > 1 else 0.0
	sharpe = float(np.mean(strategy_ret) / ret_std * np.sqrt(252.0)) if ret_std > 0 else float("nan")

	return {
		"mse": mse,
		"ic": ic,
		"rankic": rankic,
		"sign_accuracy": sign_accuracy,
		"sharpe": sharpe,
	}


def topk_backtest_metrics(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	trade_date: pd.Series,
	topk: int,
) -> dict[str, float]:
	df_bt = pd.DataFrame(
		{
			"trade_date": trade_date.to_numpy(),
			"y_true": y_true,
			"y_pred": y_pred,
		}
	)

	daily_ret: list[float] = []
	for _, group in df_bt.groupby("trade_date", sort=True):
		picked = group.nlargest(topk, "y_pred")
		if picked.empty:
			continue
		daily_ret.append(float(picked["y_true"].mean()))

	if not daily_ret:
		return {"irr": float("nan"), "sharpe": float("nan")}

	daily_ret_arr = np.asarray(daily_ret, dtype=np.float64)
	gross = 1.0 + daily_ret_arr
	if np.any(gross <= 0):
		irr = float("nan")
	else:
		irr = float(np.prod(gross) ** (252.0 / len(daily_ret_arr)) - 1.0)

	ret_std = float(np.std(daily_ret_arr, ddof=1)) if len(daily_ret_arr) > 1 else 0.0
	sharpe = float(np.mean(daily_ret_arr) / ret_std * np.sqrt(252.0)) if ret_std > 0 else float("nan")

	return {"irr": irr, "sharpe": sharpe}


def predict_split(
	model: nn.Module,
	split_set: dict[str, object],
	edge_index: torch.Tensor,
	device: torch.device,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
	x_all = split_set["x"]
	y_all = split_set["y"]
	mask_all = split_set["mask"]
	trade_dates = split_set["trade_dates"]

	if not isinstance(x_all, torch.Tensor) or not isinstance(y_all, torch.Tensor):
		raise TypeError("split keys 'x' and 'y' must be torch.Tensor")
	if not isinstance(mask_all, torch.Tensor):
		raise TypeError("split key 'mask' must be torch.Tensor")
	if not isinstance(trade_dates, list):
		raise TypeError("split key 'trade_dates' must be list")

	if x_all.shape[0] != len(trade_dates):
		raise ValueError("Number of trade dates must match split length T.")

	edge_index = edge_index.to(device)
	model.eval()
	all_true: list[float] = []
	all_pred: list[float] = []
	all_dates: list[str] = []

	with torch.no_grad():
		for t in range(x_all.shape[0]):
			x = x_all[t].to(device)
			y = y_all[t].to(device)
			mask = mask_all[t].to(device)

			pred = model(x, edge_index).squeeze(-1)
			if int(mask.sum().item()) == 0:
				continue

			y_np = y[mask].cpu().numpy().astype(np.float64)
			pred_np = pred[mask].cpu().numpy().astype(np.float64)
			all_true.extend(y_np.tolist())
			all_pred.extend(pred_np.tolist())
			all_dates.extend([str(trade_dates[t])] * len(y_np))

	return (
		np.asarray(all_true, dtype=np.float64),
		np.asarray(all_pred, dtype=np.float64),
		pd.Series(all_dates),
	)


def load_state_dict_into_model(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
	model.load_state_dict(state_dict)


def evaluate_split(
	name: str,
	model: nn.Module,
	split_set: dict[str, object],
	edge_index: torch.Tensor,
	device: torch.device,
	topk_values: list[int],
) -> list[str]:
	y_true, y_pred, trade_date = predict_split(model, split_set, edge_index, device)
	if len(y_true) == 0:
		return [f"[{name}] samples=0", f"[{name}] no valid labeled samples for evaluation"]

	metrics = regression_metrics(y_true, y_pred)
	lines = [
		f"[{name}] samples={len(y_true)}",
		f"[{name}] IC={metrics['ic']:.8f} | RankIC={metrics['rankic']:.8f} | "
		f"SignAcc={metrics['sign_accuracy']:.8f} | Sharpe={metrics['sharpe']:.8f} | "
		f"MSE={metrics['mse']:.8f}",
	]

	for k in topk_values:
		bt_metrics = topk_backtest_metrics(y_true, y_pred, trade_date, k)
		lines.append(
			f"[{name}] Top{k} IRR={bt_metrics['irr']:.8f} | Top{k} Sharpe={bt_metrics['sharpe']:.8f}"
		)

	return lines


def save_outputs(
	save_model_path: Path,
	output_log: Path,
	model: nn.Module,
	feature_cols: list[str],
	target_col: str,
	node_codes: list[str],
	edge_index: torch.Tensor,
	result_lines: list[str],
) -> None:
	save_model_path.parent.mkdir(parents=True, exist_ok=True)
	output_log.parent.mkdir(parents=True, exist_ok=True)

	ckpt = {
		"model_state_dict": model.state_dict(),
		"feature_cols": feature_cols,
		"target_col": target_col,
		"node_codes": node_codes,
		"edge_index": edge_index,
		"model": "GCN(15->32->1)",
	}
	torch.save(ckpt, save_model_path)

	with output_log.open("w", encoding="utf-8") as f:
		for line in result_lines:
			f.write(line + "\n")


def main() -> None:
	args = parse_args()
	topk_values = parse_topk_list(args.topk_list)
	device = resolve_device(args.device)

	metadata, train_set, val_set, test_set = load_graph_dataset(args.graph_data_dir)

	edge_index = metadata["edge_index"]
	node_codes = metadata["node_codes"]
	feature_cols = metadata["feature_cols"]
	target_col = metadata["target_col"]

	if not isinstance(edge_index, torch.Tensor):
		raise TypeError("metadata edge_index must be torch.Tensor")
	if not isinstance(node_codes, list):
		raise TypeError("metadata node_codes must be list")
	if not isinstance(feature_cols, list):
		raise TypeError("metadata feature_cols must be list")
	if not isinstance(target_col, str):
		raise TypeError("metadata target_col must be str")

	model = GCN(in_dim=len(feature_cols), dropout=args.dropout)
	train_logs, best_state_dict, best_val_ic, best_epoch = train(
		model=model,
		train_set=train_set,
		val_set=val_set,
		edge_index=edge_index,
		device=device,
		epochs=args.epochs,
		lr=args.lr,
		weight_decay=args.weight_decay,
		early_stopping_patience=args.early_stopping_patience,
	)
	load_state_dict_into_model(model, best_state_dict)

	result_lines = [
		"GCN Regression (fixed year split, prebuilt graph dataset)",
		"train years: [2019, 2020, 2021, 2022, 2023]",
		"val years: [2024]",
		"test years: [2025]",
		f"target: {target_col}",
		f"epochs: {args.epochs}",
		f"lr: {args.lr}",
		f"weight_decay: {args.weight_decay}",
		f"dropout: {args.dropout}",
		f"early_stopping_patience: {args.early_stopping_patience}",
		f"topk_list: {topk_values}",
		f"features ({len(feature_cols)}): {feature_cols}",
		f"best_epoch: {best_epoch}",
		f"best_val_ic: {best_val_ic:.8f}",
	]
	result_lines.extend(train_logs)
	result_lines.extend(
		evaluate_split("train", model, train_set, edge_index, device, topk_values)
	)
	result_lines.extend(
		evaluate_split("validation", model, val_set, edge_index, device, topk_values)
	)
	result_lines.extend(
		evaluate_split("test", model, test_set, edge_index, device, topk_values)
	)

	save_outputs(
		save_model_path=args.save_model_path,
		output_log=args.output_log,
		model=model,
		feature_cols=feature_cols,
		target_col=target_col,
		node_codes=node_codes,
		edge_index=edge_index,
		result_lines=result_lines,
	)

	train_steps = int(train_set["x"].shape[0])
	val_steps = int(val_set["x"].shape[0])
	test_steps = int(test_set["x"].shape[0])
	print(f"Training finished. Train steps: {train_steps}")
	print(f"Val steps: {val_steps}")
	print(f"Test steps: {test_steps}")
	print(f"Model saved to: {args.save_model_path}")
	print(f"Log saved to: {args.output_log}")
	for line in result_lines:
		print(line)


if __name__ == "__main__":
	main()
