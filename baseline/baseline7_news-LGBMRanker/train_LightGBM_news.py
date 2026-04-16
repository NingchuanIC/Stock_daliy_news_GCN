from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
	import lightgbm as lgb
except ImportError as exc:
	raise ImportError(
		"lightgbm is required. Install it with: pip install lightgbm"
	) from exc


TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
VAL_YEARS = [2024]
TEST_YEARS = [2025]
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
	"news_cnt_1d",
	"news_cnt_3d",
	"news_cnt_7d",
	"news_sent_mean_1d",
	"news_sent_mean_3d",
	"news_sent_mean_7d",
	"news_sent_std_7d",
	"news_sent_max_3d",
	"news_sent_min_3d",
	"news_pos_cnt_3d",
	"news_neg_cnt_3d",
	"news_risk_cnt_7d",
	"days_since_last_news",
]

PRICE_FEATURE_COLS = [
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
		description="Train and evaluate a LightGBM model on monthly parquet features."
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(__file__).resolve().parent,
		help="Directory where result txt files will be written.",
	)
	parser.add_argument(
		"--n-estimators",
		type=int,
		default=300,
		help="Number of boosting rounds.",
	)
	parser.add_argument(
		"--learning-rate",
		type=float,
		default=0.05,
		help="Boosting learning rate.",
	)
	parser.add_argument(
		"--num-leaves",
		type=int,
		default=63,
		help="Maximum tree leaves for each weak learner.",
	)
	parser.add_argument(
		"--feature-fraction",
		type=float,
		default=0.9,
		help="Randomly select this fraction of features on each iteration.",
	)
	parser.add_argument(
		"--bagging-fraction",
		type=float,
		default=0.9,
		help="Randomly select this fraction of rows on each iteration.",
	)
	parser.add_argument(
		"--bagging-freq",
		type=int,
		default=1,
		help="Bagging frequency. 0 disables row bagging.",
	)
	parser.add_argument(
		"--min-child-samples",
		type=int,
		default=100,
		help="Minimum number of data points in a leaf.",
	)
	parser.add_argument(
		"--reg-alpha",
		type=float,
		default=0.0,
		help="L1 regularization.",
	)
	parser.add_argument(
		"--reg-lambda",
		type=float,
		default=1.0,
		help="L2 regularization.",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Random seed for reproducibility.",
	)
	parser.add_argument(
		"--topk-list",
		type=str,
		default="10,30,50",
		help="Comma-separated Top-k list for cross-sectional backtest metrics (IRR/Sharpe).",
	)
	return parser.parse_args()


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
	if "trade_date" in df.columns:
		df = df.sort_values(["trade_date", "ts_code"], kind="stable").reset_index(drop=True)
	return df


def build_features(
	df: pd.DataFrame, target_col: str
) -> tuple[np.ndarray, np.ndarray, list[str], pd.Series | None]:
	if target_col not in df.columns:
		raise KeyError(f"Target column '{target_col}' not found in dataset columns.")

	missing = [col for col in PRICE_FEATURE_COLS if col not in df.columns]
	if missing:
		raise KeyError(f"Dataset is missing required features: {missing}")

	feature_cols = [col for col in FEATURE_COLS if col in df.columns]

	selected_cols = feature_cols + [target_col]
	if "trade_date" in df.columns:
		selected_cols.append("trade_date")

	selected = df[selected_cols].replace([np.inf, -np.inf], np.nan).dropna()
	x = selected[feature_cols].to_numpy(dtype=np.float64)
	y = selected[target_col].to_numpy(dtype=np.float64)
	trade_date = selected["trade_date"] if "trade_date" in selected.columns else None
	return x, y, feature_cols, trade_date


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

	# Keep insertion order while removing duplicates.
	return list(dict.fromkeys(values))


def fit_lightgbm_regression(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> lgb.LGBMRegressor:
	model = lgb.LGBMRegressor(
		objective="regression",
		n_estimators=args.n_estimators,
		learning_rate=args.learning_rate,
		num_leaves=args.num_leaves,
		feature_fraction=args.feature_fraction,
		bagging_fraction=args.bagging_fraction,
		bagging_freq=args.bagging_freq,
		min_child_samples=args.min_child_samples,
		reg_alpha=args.reg_alpha,
		reg_lambda=args.reg_lambda,
		random_state=args.random_state,
		n_jobs=-1,
	)
	model.fit(x, y)
	return model


def predict(model: lgb.LGBMRegressor, x: np.ndarray) -> np.ndarray:
	return np.asarray(model.predict(x), dtype=np.float64)


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
	trade_date: pd.Series | None,
	topk: int,
) -> dict[str, float]:
	if trade_date is None:
		return {"irr": float("nan"), "sharpe": float("nan")}

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


def evaluate_split(
	name: str,
	model: lgb.LGBMRegressor,
	x: np.ndarray,
	y: np.ndarray,
	trade_date: pd.Series | None,
	topk_values: list[int],
) -> list[str]:
	y_pred = predict(model, x)
	metrics = regression_metrics(y, y_pred)
	lines = [
		f"[{name}] samples={len(y)}",
		f"[{name}] IC={metrics['ic']:.8f} | RankIC={metrics['rankic']:.8f} | "
		f"SignAcc={metrics['sign_accuracy']:.8f} | Sharpe={metrics['sharpe']:.8f} | "
		f"MSE={metrics['mse']:.8f}",
	]
	for k in topk_values:
		bt_metrics = topk_backtest_metrics(y, y_pred, trade_date, k)
		lines.append(
			f"[{name}] Top{k} IRR={bt_metrics['irr']:.8f} | Top{k} Sharpe={bt_metrics['sharpe']:.8f}"
		)

	return lines


def run_experiment(
	args: argparse.Namespace,
	label: str,
	dataset_root: Path,
	target_col: str,
	output_path: Path,
	topk_values: list[int],
) -> None:
	train_df = load_years(dataset_root, TRAIN_YEARS)
	val_df = load_years(dataset_root, VAL_YEARS)
	test_df = load_years(dataset_root, TEST_YEARS)

	x_train, y_train, feature_cols, trade_date_train = build_features(train_df, target_col)
	x_val, y_val, _, trade_date_val = build_features(val_df, target_col)
	x_test, y_test, _, trade_date_test = build_features(test_df, target_col)

	model = fit_lightgbm_regression(x_train, y_train, args)

	result_lines = [
		"LightGBM Regression (no shuffling, fixed year split)",
		f"train years: {TRAIN_YEARS}",
		f"val years: {VAL_YEARS}",
		f"test years: {TEST_YEARS}",
		f"target: {target_col}",
		f"n_estimators: {args.n_estimators}",
		f"learning_rate: {args.learning_rate}",
		f"num_leaves: {args.num_leaves}",
		f"feature_fraction: {args.feature_fraction}",
		f"bagging_fraction: {args.bagging_fraction}",
		f"bagging_freq: {args.bagging_freq}",
		f"min_child_samples: {args.min_child_samples}",
		f"reg_alpha: {args.reg_alpha}",
		f"reg_lambda: {args.reg_lambda}",
		f"random_state: {args.random_state}",
		f"topk_list: {topk_values}",
		f"features ({len(feature_cols)}): {feature_cols}",
	]

	result_lines.extend(evaluate_split("train", model, x_train, y_train, trade_date_train, topk_values))
	result_lines.extend(evaluate_split("validation", model, x_val, y_val, trade_date_val, topk_values))
	result_lines.extend(evaluate_split("test", model, x_test, y_test, trade_date_test, topk_values))

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")

	print(f"===== {label} =====")
	for line in result_lines:
		print(line)
	print(f"Results saved to: {output_path}")


def main() -> None:
	args = parse_args()
	topk_values = parse_topk_list(args.topk_list)

	data_root = Path(__file__).resolve().parents[2] / "data"
	experiments = [
		(
			"ret5",
			data_root / "build_feature_news" / "dataset_by_month",
			"future_ret_5",
			args.output_dir / "lightgbm_results_ret5.txt",
		),
		(
			"ret1",
			data_root / "build_feature_news_ret1" / "dataset_by_month",
			"future_ret_1",
			args.output_dir / "lightgbm_results_ret1.txt",
		),
	]

	for label, dataset_root, target_col, output_path in experiments:
		run_experiment(
			args=args,
			label=label,
			dataset_root=dataset_root,
			target_col=target_col,
			output_path=output_path,
			topk_values=topk_values,
		)


if __name__ == "__main__":
	main()
