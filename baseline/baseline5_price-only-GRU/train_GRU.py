from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
	import torch
	from torch import nn
	from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
	raise ImportError(
		"pytorch is required. Install it with: pip install torch"
	) from exc


TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
VAL_YEARS = [2024]
TEST_YEARS = [2025]
FEATURE_COLS_15 = [
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train and evaluate a price-only GRU model on monthly parquet data."
	)
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=Path(__file__).resolve().parents[1] / "build_feature" / "dataset_by_month",
		help="Root folder containing yearly subfolders of monthly parquet files.",
	)
	parser.add_argument(
		"--target",
		type=str,
		default="future_ret_5",
		help="Target column used for supervised learning.",
	)
	parser.add_argument(
		"--task",
		type=str,
		choices=["regression", "classification"],
		default="regression",
		help="Training task. Classification uses binary target from future return sign.",
	)
	parser.add_argument("--seq-len", type=int, default=30, help="Sequence length T.")
	parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size in RNN layers.")
	parser.add_argument("--num-layers", type=int, default=2, help="Number of stacked RNN layers.")
	parser.add_argument("--dropout", type=float, default=0.1, help="Dropout in RNN/MLP.")
	parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size.")
	parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
	parser.add_argument(
		"--early-stopping-patience",
		type=int,
		default=20,
		help="Stop training when validation IC does not improve for this many epochs.",
	)
	parser.add_argument(
		"--early-stopping-min-delta",
		type=float,
		default=0.0,
		help="Minimum increase in validation IC to count as improvement.",
	)
	parser.add_argument(
		"--optimizer",
		type=str,
		choices=["adam", "rmsprop"],
		default="adam",
		help="Optimizer choice. HATS-style config usually uses rmsprop for RNN.",
	)
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
	parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
	parser.add_argument(
		"--feature-norm",
		type=str,
		choices=["max", "zscore", "none"],
		default="max",
		help="Per-sequence normalization method.",
	)
	parser.add_argument(
		"--topk-list",
		type=str,
		default="10,30,50",
		help="Comma-separated Top-k list for cross-sectional backtest metrics (IRR/Sharpe).",
	)
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument(
		"--device",
		type=str,
		choices=["cuda", "cpu", "auto"],
		default="cuda",
		help="Execution device. Default is cuda and will fail fast if CUDA is unavailable.",
	)
	parser.add_argument(
		"--gpu-id",
		type=int,
		default=0,
		help="CUDA device id when using GPU.",
	)
	parser.add_argument(
		"--num-workers",
		type=int,
		default=2,
		help="DataLoader worker processes.",
	)
	parser.add_argument(
		"--amp",
		action="store_true",
		help="Enable automatic mixed precision on CUDA for faster training.",
	)
	parser.add_argument(
		"--load-model-path",
		type=Path,
		default=None,
		help="Optional checkpoint path to load model weights before training/evaluation.",
	)
	parser.add_argument(
		"--save-model-path",
		type=Path,
		default=Path(__file__).resolve().parent / "gru_model.pt",
		help="Checkpoint path used to save model weights after training.",
	)
	parser.add_argument(
		"--output-file",
		type=Path,
		default=Path(__file__).resolve().parent / "gru_results.txt",
		help="Path to the txt file where training/evaluation results will be written.",
	)
	return parser.parse_args()


def load_model_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> str:
	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

	checkpoint = torch.load(checkpoint_path, map_location=device)
	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		model.load_state_dict(checkpoint["model_state_dict"])
		epoch = checkpoint.get("epoch", "unknown")
		return f"loaded checkpoint from {checkpoint_path} (epoch={epoch})"

	if isinstance(checkpoint, dict):
		model.load_state_dict(checkpoint)
		return f"loaded state_dict from {checkpoint_path}"

	raise ValueError("Unsupported checkpoint format. Expected state_dict or dict with model_state_dict.")


def save_model_checkpoint(
	model: nn.Module,
	checkpoint_path: Path,
	args: argparse.Namespace,
	feature_cols: list[str],
	epochs_ran: int,
) -> str:
	checkpoint_path = Path(checkpoint_path)
	if checkpoint_path.suffix == "" or checkpoint_path.is_dir():
		checkpoint_path = checkpoint_path / "gru_model.pt"
	checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
	checkpoint = {
		"model_state_dict": model.state_dict(),
		"model_type": "gru",
		"feature_cols": feature_cols,
		"config": {
			"seq_len": args.seq_len,
			"hidden_size": args.hidden_size,
			"num_layers": args.num_layers,
			"dropout": args.dropout,
			"task": args.task,
			"target": args.target,
		},
		"epoch": epochs_ran,
	}
	torch.save(checkpoint, checkpoint_path)
	return f"saved checkpoint to {checkpoint_path}"


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = False
	torch.backends.cudnn.benchmark = True
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True


def resolve_device(args: argparse.Namespace) -> torch.device:
	if args.device == "cpu":
		return torch.device("cpu")

	has_cuda = torch.cuda.is_available()
	if args.device == "cuda" and not has_cuda:
		raise RuntimeError(
			"--device=cuda was requested but CUDA is not available in current PyTorch runtime."
		)

	if has_cuda:
		torch.cuda.set_device(args.gpu_id)
		return torch.device(f"cuda:{args.gpu_id}")

	return torch.device("cpu")


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


def load_years(dataset_root: Path, years: list[int]) -> pd.DataFrame:
	frames: list[pd.DataFrame] = []
	for year in years:
		year_dir = dataset_root / str(year)
		if not year_dir.exists():
			raise FileNotFoundError(f"Year directory not found: {year_dir}")

		parquet_files = sorted(year_dir.glob("*.parquet"))
		if not parquet_files:
			raise FileNotFoundError(f"No parquet files found in: {year_dir}")

		for parquet_path in parquet_files:
			frames.append(pd.read_parquet(parquet_path))

	df = pd.concat(frames, ignore_index=True)
	need_cols = ["ts_code", "trade_date"]
	missing = [c for c in need_cols if c not in df.columns]
	if missing:
		raise KeyError(f"Required columns missing: {missing}")

	if not pd.api.types.is_datetime64_any_dtype(df["trade_date"]):
		df["trade_date"] = pd.to_datetime(
			df["trade_date"].astype(str).str.slice(0, 8),
			format="%Y%m%d",
			errors="coerce",
		)

	df = df.dropna(subset=["trade_date", "ts_code"])  # keep aligned panel only
	df = df.sort_values(["ts_code", "trade_date"], kind="stable").reset_index(drop=True)
	return df


def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	if "close" not in out.columns:
		return out

	for col in ["open", "high", "low", "close", "vol"]:
		if col in out.columns:
			out[col] = pd.to_numeric(out[col], errors="coerce")

	group = out.groupby("ts_code", sort=False)
	out["ma5"] = group["close"].transform(lambda s: s.rolling(5, min_periods=5).mean())
	out["ma10"] = group["close"].transform(lambda s: s.rolling(10, min_periods=10).mean())
	out["ma20"] = group["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
	out["ma30"] = group["close"].transform(lambda s: s.rolling(30, min_periods=30).mean())

	# RSR-style normalized close proxy using trailing max.
	out["close_norm"] = group["close"].transform(lambda s: s / s.rolling(30, min_periods=1).max())
	return out


def choose_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
	_ = target_col
	missing = [c for c in FEATURE_COLS_15 if c not in df.columns]
	if missing:
		raise KeyError(f"Dataset is missing required 15 features: {missing}")
	return FEATURE_COLS_15.copy()


def normalize_sequence(x_seq: np.ndarray, mode: str) -> np.ndarray:
	if mode == "none":
		return x_seq

	if mode == "max":
		den = np.max(np.abs(x_seq), axis=0, keepdims=True)
		den = np.where(den < 1e-8, 1.0, den)
		return x_seq / den

	if mode == "zscore":
		mu = np.mean(x_seq, axis=0, keepdims=True)
		std = np.std(x_seq, axis=0, keepdims=True)
		std = np.where(std < 1e-8, 1.0, std)
		return (x_seq - mu) / std

	raise ValueError(f"Unknown normalization mode: {mode}")


@dataclass
class SequencePanel:
	x: np.ndarray
	y_train: np.ndarray
	y_ret: np.ndarray
	trade_date: np.ndarray


def build_sequence_panel(
	df: pd.DataFrame,
	feature_cols: list[str],
	target_col: str,
	seq_len: int,
	norm_mode: str,
	task: str,
) -> SequencePanel:
	if target_col not in df.columns:
		raise KeyError(f"Target column '{target_col}' not found in dataset columns.")

	work = df[["ts_code", "trade_date", *feature_cols, target_col]].copy()
	work[feature_cols] = work[feature_cols].replace([np.inf, -np.inf], np.nan)
	work[target_col] = pd.to_numeric(work[target_col], errors="coerce")

	seq_x: list[np.ndarray] = []
	seq_y_train: list[float] = []
	seq_y_ret: list[float] = []
	seq_dates: list[pd.Timestamp] = []

	for _, g in work.groupby("ts_code", sort=False):
		g = g.sort_values("trade_date", kind="stable")
		x_raw = g[feature_cols].to_numpy(dtype=np.float32)
		y_raw = g[target_col].to_numpy(dtype=np.float32)
		dates = g["trade_date"].to_numpy()

		n = len(g)
		if n < seq_len:
			continue

		for end_idx in range(seq_len - 1, n):
			start_idx = end_idx - seq_len + 1
			x_seq = x_raw[start_idx : end_idx + 1]
			y_ret = float(y_raw[end_idx])

			if np.isnan(y_ret) or np.isinf(y_ret):
				continue
			if np.isnan(x_seq).any() or np.isinf(x_seq).any():
				continue

			x_seq = normalize_sequence(x_seq, norm_mode)
			seq_x.append(x_seq.astype(np.float32, copy=False))
			seq_y_ret.append(y_ret)
			if task == "classification":
				seq_y_train.append(1.0 if y_ret > 0.0 else 0.0)
			else:
				seq_y_train.append(y_ret)
			seq_dates.append(dates[end_idx])

	if not seq_x:
		raise ValueError("No valid sequences were built. Check feature availability / seq_len.")

	return SequencePanel(
		x=np.asarray(seq_x, dtype=np.float32),
		y_train=np.asarray(seq_y_train, dtype=np.float32),
		y_ret=np.asarray(seq_y_ret, dtype=np.float32),
		trade_date=np.asarray(seq_dates),
	)


class SequenceDataset(Dataset):
	def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
		self.x = torch.from_numpy(x)
		self.y = torch.from_numpy(y).unsqueeze(1)

	def __len__(self) -> int:
		return int(self.x.shape[0])

	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		return self.x[idx], self.y[idx]


class RNNRegressor(nn.Module):
	def __init__(
		self,
		input_size: int,
		hidden_size: int,
		num_layers: int,
		dropout: float,
	) -> None:
		super().__init__()
		self.rnn = nn.GRU(
			input_size=input_size,
			hidden_size=hidden_size,
			num_layers=num_layers,
			dropout=dropout if num_layers > 1 else 0.0,
			batch_first=True,
		)
		self.head = nn.Sequential(
			nn.Linear(hidden_size, 128),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out, _ = self.rnn(x)
		last = out[:, -1, :]
		return self.head(last)


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
	if args.optimizer == "adam":
		return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	return torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_loss(task: str) -> nn.Module:
	if task == "classification":
		return nn.BCEWithLogitsLoss()
	return nn.MSELoss()


def run_epoch(
	model: nn.Module,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	loss_fn: nn.Module,
	device: torch.device,
	use_amp: bool,
	scaler: torch.amp.GradScaler | None,
) -> float:
	model.train()
	total_loss = 0.0
	total_cnt = 0
	amp_enabled = use_amp and device.type == "cuda"
	for xb, yb in loader:
		xb = xb.to(device, non_blocking=True)
		yb = yb.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)
		with torch.amp.autocast("cuda", enabled=amp_enabled):
			pred = model(xb)
			loss = loss_fn(pred, yb)

		if amp_enabled and scaler is not None:
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()

		batch = int(xb.shape[0])
		total_loss += float(loss.item()) * batch
		total_cnt += batch

	return total_loss / max(total_cnt, 1)


@torch.no_grad()
def infer_scores(
	model: nn.Module,
	x: np.ndarray,
	batch_size: int,
	device: torch.device,
	task: str,
	use_amp: bool,
) -> np.ndarray:
	model.eval()
	loader = DataLoader(
		torch.from_numpy(x),
		batch_size=batch_size,
		shuffle=False,
		pin_memory=device.type == "cuda",
	)
	preds: list[np.ndarray] = []
	amp_enabled = use_amp and device.type == "cuda"
	for xb in loader:
		xb = xb.to(device, non_blocking=True)
		with torch.amp.autocast("cuda", enabled=amp_enabled):
			out = model(xb).squeeze(1)
		score = torch.sigmoid(out) - 0.5 if task == "classification" else out
		preds.append(score.detach().cpu().numpy())
	return np.concatenate(preds).astype(np.float64, copy=False)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
	residual = y_true - y_pred
	mse = float(np.mean(residual**2))
	sign_accuracy = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

	ic = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) > 1 else float("nan")
	pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=np.float64)
	true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=np.float64)
	rankic = float(np.corrcoef(pred_rank, true_rank)[0, 1]) if len(y_true) > 1 else float("nan")

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
	trade_date: np.ndarray,
	topk: int,
) -> dict[str, float]:
	df_bt = pd.DataFrame(
		{
			"trade_date": trade_date,
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


def evaluate_split(
	name: str,
	model: nn.Module,
	panel: SequencePanel,
	batch_size: int,
	device: torch.device,
	task: str,
	use_amp: bool,
	topk_values: list[int],
) -> list[str]:
	y_pred = infer_scores(
		model,
		panel.x,
		batch_size=batch_size,
		device=device,
		task=task,
		use_amp=use_amp,
	)
	metrics = regression_metrics(panel.y_ret.astype(np.float64), y_pred)

	lines = [
		f"[{name}] samples={len(panel.y_ret)}",
		f"[{name}] IC={metrics['ic']:.8f} | RankIC={metrics['rankic']:.8f} | "
		f"SignAcc={metrics['sign_accuracy']:.8f} | Sharpe={metrics['sharpe']:.8f} | "
		f"MSE={metrics['mse']:.8f}",
	]
	for k in topk_values:
		bt_metrics = topk_backtest_metrics(panel.y_ret.astype(np.float64), y_pred, panel.trade_date, k)
		lines.append(
			f"[{name}] Top{k} IRR={bt_metrics['irr']:.8f} | Top{k} Sharpe={bt_metrics['sharpe']:.8f}"
		)
	return lines


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	topk_values = parse_topk_list(args.topk_list)

	train_df = _add_price_features(load_years(args.dataset_root, TRAIN_YEARS))
	val_df = _add_price_features(load_years(args.dataset_root, VAL_YEARS))
	test_df = _add_price_features(load_years(args.dataset_root, TEST_YEARS))

	feature_cols = choose_feature_columns(train_df, args.target)
	train_panel = build_sequence_panel(
		train_df, feature_cols, args.target, args.seq_len, args.feature_norm, args.task
	)
	val_panel = build_sequence_panel(
		val_df, feature_cols, args.target, args.seq_len, args.feature_norm, args.task
	)
	test_panel = build_sequence_panel(
		test_df, feature_cols, args.target, args.seq_len, args.feature_norm, args.task
	)

	device = resolve_device(args)
	model = RNNRegressor(
		input_size=train_panel.x.shape[2],
		hidden_size=args.hidden_size,
		num_layers=args.num_layers,
		dropout=args.dropout,
	).to(device)

	load_status = "not loaded"
	if args.load_model_path is not None:
		load_status = load_model_checkpoint(model, args.load_model_path, device)

	train_loader = DataLoader(
		SequenceDataset(train_panel.x, train_panel.y_train),
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=device.type == "cuda",
	)
	optimizer = build_optimizer(model, args)
	loss_fn = build_loss(args.task)
	use_amp = args.amp and device.type == "cuda"
	scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

	history_lines: list[str] = []
	epochs_ran = 0
	best_epoch = 0
	best_val_ic = float("-inf")
	best_state_dict: dict[str, torch.Tensor] | None = None
	no_improve_count = 0
	for epoch in range(1, args.epochs + 1):
		train_loss = run_epoch(model, train_loader, optimizer, loss_fn, device, use_amp, scaler)
		val_pred = infer_scores(model, val_panel.x, args.batch_size, device, args.task, use_amp)
		val_loss = float(np.mean((val_pred - val_panel.y_ret.astype(np.float64)) ** 2))
		val_metrics = regression_metrics(val_panel.y_ret.astype(np.float64), val_pred)
		val_ic = val_metrics["ic"]
		epoch_line = (
			f"[epoch {epoch:03d}/{args.epochs:03d}] train_loss={train_loss:.8f} | "
			f"val_proxy_loss={val_loss:.8f} | val_ic={val_ic:.8f}"
		)
		history_lines.append(epoch_line)
		print(epoch_line, flush=True)
		epochs_ran = epoch

		if np.isfinite(val_ic) and val_ic > best_val_ic + args.early_stopping_min_delta:
			best_val_ic = val_ic
			best_epoch = epoch
			best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
			no_improve_count = 0
		else:
			no_improve_count += 1

		if no_improve_count >= args.early_stopping_patience:
			es_line = (
				f"[early_stopping] stop at epoch {epoch:03d} | "
				f"best_epoch={best_epoch:03d} | best_val_ic={best_val_ic:.8f}"
			)
			history_lines.append(es_line)
			print(es_line, flush=True)
			break

	if best_state_dict is not None:
		model.load_state_dict(best_state_dict)
		restore_line = (
			f"[early_stopping] restored best model at epoch {best_epoch:03d} "
			f"with val_ic={best_val_ic:.8f}"
		)
		history_lines.append(restore_line)
		print(restore_line, flush=True)

	save_status = save_model_checkpoint(model, args.save_model_path, args, feature_cols, epochs_ran)

	result_lines = [
		"Price-only GRU Baseline (no shuffling, fixed year split)",
		f"train years: {TRAIN_YEARS}",
		f"val years: {VAL_YEARS}",
		f"test years: {TEST_YEARS}",
		f"task: {args.task}",
		f"target: {args.target}",
		"model_type: gru",
		f"seq_len: {args.seq_len}",
		f"hidden_size: {args.hidden_size}",
		f"num_layers: {args.num_layers}",
		f"dropout: {args.dropout}",
		f"optimizer: {args.optimizer}",
		f"lr: {args.lr}",
		f"batch_size: {args.batch_size}",
		f"epochs: {args.epochs}",
		f"epochs_ran: {epochs_ran}",
		f"early_stopping_patience: {args.early_stopping_patience}",
		f"early_stopping_min_delta: {args.early_stopping_min_delta}",
		f"best_epoch: {best_epoch}",
		f"best_val_ic: {best_val_ic:.8f}",
		f"feature_norm: {args.feature_norm}",
		f"seed: {args.seed}",
		f"device_arg: {args.device}",
		f"gpu_id: {args.gpu_id}",
		f"device: {device}",
		f"amp: {use_amp}",
		f"num_workers: {args.num_workers}",
		f"load_model_path: {args.load_model_path}",
		f"save_model_path: {args.save_model_path}",
		f"load_status: {load_status}",
		f"save_status: {save_status}",
		f"topk_list: {topk_values}",
		f"features ({len(feature_cols)}): {feature_cols}",
		f"train sequences: {len(train_panel.y_ret)}",
		f"val sequences: {len(val_panel.y_ret)}",
		f"test sequences: {len(test_panel.y_ret)}",
	]
	if device.type == "cuda":
		result_lines.append(f"gpu_name: {torch.cuda.get_device_name(device)}")
	result_lines.extend(history_lines)
	result_lines.extend(
		evaluate_split("train", model, train_panel, args.batch_size, device, args.task, use_amp, topk_values)
	)
	result_lines.extend(
		evaluate_split(
			"validation", model, val_panel, args.batch_size, device, args.task, use_amp, topk_values
		)
	)
	result_lines.extend(
		evaluate_split("test", model, test_panel, args.batch_size, device, args.task, use_amp, topk_values)
	)

	output_path = args.output_file
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")

	for line in result_lines:
		print(line)
	print(f"Results saved to: {output_path}")


if __name__ == "__main__":
	main()
