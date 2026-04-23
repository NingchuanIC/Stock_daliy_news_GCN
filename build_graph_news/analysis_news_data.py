from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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

NEWS_FEATURE_COLS = [
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


def parse_args() -> argparse.Namespace:
	repo_root = Path(__file__).resolve().parents[1]
	parser = argparse.ArgumentParser(
		description="Analyze source price+news features used by graph construction."
	)
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=repo_root / "data" / "build_feature_news" / "dataset_by_month",
		help="Root folder containing year/month parquet source files.",
	)
	parser.add_argument(
		"--graph-data-dir",
		type=Path,
		default=repo_root / "build_graph_news" / "graph_dataset",
		help="Graph dataset directory containing metadata.pt/train.pt/val.pt/test.pt.",
	)
	parser.add_argument(
		"--target-col",
		type=str,
		default="future_ret_5",
		help="Target column in source parquet files.",
	)
	parser.add_argument(
		"--out-json",
		type=Path,
		default=repo_root / "build_graph_news" / "analysis_news_data_report.json",
		help="Path to write analysis report JSON.",
	)
	return parser.parse_args()


def iter_parquet_files(dataset_root: Path) -> list[Path]:
	if not dataset_root.exists():
		raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
	files = sorted(dataset_root.glob("*/*.parquet"))
	if not files:
		raise FileNotFoundError(f"No parquet files under: {dataset_root}")
	return files


def _json_safe(value: object) -> object:
	if isinstance(value, (np.floating,)):
		v = float(value)
		if np.isnan(v) or np.isinf(v):
			return None
		return v
	if isinstance(value, (np.integer,)):
		return int(value)
	if isinstance(value, pd.Timestamp):
		return value.strftime("%Y-%m-%d")
	return value


def _series_basic_stats(series: pd.Series) -> dict[str, object]:
	numeric = pd.to_numeric(series, errors="coerce")
	valid = numeric.dropna()
	if valid.empty:
		return {
			"count": 0,
			"mean": None,
			"std": None,
			"min": None,
			"p01": None,
			"p05": None,
			"p50": None,
			"p95": None,
			"p99": None,
			"max": None,
		}

	q = valid.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
	return {
		"count": int(valid.shape[0]),
		"mean": _json_safe(valid.mean()),
		"std": _json_safe(valid.std(ddof=1)),
		"min": _json_safe(valid.min()),
		"p01": _json_safe(q.loc[0.01]),
		"p05": _json_safe(q.loc[0.05]),
		"p50": _json_safe(q.loc[0.5]),
		"p95": _json_safe(q.loc[0.95]),
		"p99": _json_safe(q.loc[0.99]),
		"max": _json_safe(valid.max()),
	}


def _series_iqr_outliers(series: pd.Series) -> tuple[int, float, float, float]:
	numeric = pd.to_numeric(series, errors="coerce")
	valid = numeric.dropna()
	if valid.empty:
		return 0, float("nan"), float("nan"), float("nan")

	q1 = float(valid.quantile(0.25))
	q3 = float(valid.quantile(0.75))
	iqr = q3 - q1
	if iqr == 0:
		return 0, q1, q3, iqr

	lower = q1 - 1.5 * iqr
	upper = q3 + 1.5 * iqr
	outlier_count = int(((valid < lower) | (valid > upper)).sum())
	return outlier_count, q1, q3, iqr


def analyze_source_dataset(
	dataset_root: Path,
	feature_cols: list[str],
	target_col: str,
) -> dict[str, object]:
	files = iter_parquet_files(dataset_root)
	needed_cols = ["ts_code", "trade_date"] + feature_cols + [target_col]

	frames: list[pd.DataFrame] = []
	for file_path in files:
		df = pd.read_parquet(file_path)
		missing = [c for c in needed_cols if c not in df.columns]
		if missing:
			raise KeyError(f"{file_path} missing columns: {missing}")
		frames.append(df[needed_cols].copy())

	panel = pd.concat(frames, ignore_index=True)
	panel["trade_date"] = pd.to_datetime(
		panel["trade_date"].astype(str).str.slice(0, 8),
		format="%Y%m%d",
		errors="coerce",
	)

	numeric_cols = feature_cols + [target_col]
	for col in numeric_cols:
		panel[col] = pd.to_numeric(panel[col], errors="coerce")

	total_rows = int(panel.shape[0])
	date_min = panel["trade_date"].min()
	date_max = panel["trade_date"].max()

	missing_by_col: dict[str, dict[str, object]] = {}
	for col in needed_cols:
		missing_count = int(panel[col].isna().sum())
		missing_by_col[col] = {
			"missing_count": missing_count,
			"missing_ratio": float(missing_count / total_rows) if total_rows > 0 else 0.0,
		}

	row_invalid_any = int(panel[numeric_cols].isna().any(axis=1).sum())
	row_invalid_ratio = float(row_invalid_any / total_rows) if total_rows > 0 else 0.0

	outliers_by_col: dict[str, dict[str, object]] = {}
	basic_stats_by_col: dict[str, dict[str, object]] = {}
	for col in numeric_cols:
		outlier_count, q1, q3, iqr = _series_iqr_outliers(panel[col])
		valid_count = int(panel[col].notna().sum())
		outliers_by_col[col] = {
			"method": "IQR_1.5",
			"outlier_count": outlier_count,
			"valid_count": valid_count,
			"outlier_ratio_among_valid": float(outlier_count / valid_count)
			if valid_count > 0
			else 0.0,
			"q1": _json_safe(q1),
			"q3": _json_safe(q3),
			"iqr": _json_safe(iqr),
		}
		basic_stats_by_col[col] = _series_basic_stats(panel[col])

	rows_by_year = (
		panel.assign(year=panel["trade_date"].dt.year)
		.groupby("year", dropna=True)
		.size()
		.astype(int)
		.to_dict()
	)

	news_count_cols = [
		"news_cnt_1d",
		"news_cnt_3d",
		"news_cnt_7d",
		"news_pos_cnt_3d",
		"news_neg_cnt_3d",
		"news_risk_cnt_7d",
	]
	news_sent_cols = [
		"news_sent_mean_1d",
		"news_sent_mean_3d",
		"news_sent_mean_7d",
		"news_sent_max_3d",
		"news_sent_min_3d",
	]
	news_default_fill_ratios = {
		"news_sent_mean_1d_eq_0_732936": float(
			(panel["news_sent_mean_1d"] == 0.732936).mean()
		),
		"news_sent_mean_3d_eq_0_732936": float(
			(panel["news_sent_mean_3d"] == 0.732936).mean()
		),
		"news_sent_mean_7d_eq_0_732936": float(
			(panel["news_sent_mean_7d"] == 0.732936).mean()
		),
		"news_sent_max_3d_eq_0_732936": float(
			(panel["news_sent_max_3d"] == 0.732936).mean()
		),
		"news_sent_min_3d_eq_0_732936": float(
			(panel["news_sent_min_3d"] == 0.732936).mean()
		),
		"days_since_last_news_eq_9999": float(
			(panel["days_since_last_news"] == 9999).mean()
		),
	}

	news_integrity_checks = {
		"sent_out_of_0_1_rows": int(
			((panel[news_sent_cols] < 0) | (panel[news_sent_cols] > 1)).any(axis=1).sum()
		),
		"count_negative_rows": int((panel[news_count_cols] < 0).any(axis=1).sum()),
		"count_non_integer_rows": int(
			(panel[news_count_cols].round() != panel[news_count_cols]).any(axis=1).sum()
		),
		"news_sent_std_7d_negative_rows": int((panel["news_sent_std_7d"] < 0).sum()),
		"news_sent_max_lt_min_rows": int(
			(panel["news_sent_max_3d"] < panel["news_sent_min_3d"]).sum()
		),
		"news_sent_mean_3d_outside_min_max_rows": int(
			(
				(panel["news_sent_mean_3d"] < panel["news_sent_min_3d"])
				| (panel["news_sent_mean_3d"] > panel["news_sent_max_3d"])
			).sum()
		),
	}

	return {
		"dataset_root": str(dataset_root),
		"file_count": int(len(files)),
		"total_rows": total_rows,
		"date_min": _json_safe(date_min),
		"date_max": _json_safe(date_max),
		"rows_by_year": {str(k): int(v) for k, v in rows_by_year.items()},
		"missing_by_col": missing_by_col,
		"rows_with_any_missing_in_model_inputs": {
			"count": row_invalid_any,
			"ratio": row_invalid_ratio,
		},
		"news_feature_focus": {
			"columns": NEWS_FEATURE_COLS,
			"default_fill_ratios": news_default_fill_ratios,
			"integrity_checks": news_integrity_checks,
		},
		"outliers_by_col": outliers_by_col,
		"basic_stats_by_col": basic_stats_by_col,
	}


def analyze_graph_dataset(graph_data_dir: Path) -> dict[str, object]:
	meta_path = graph_data_dir / "metadata.pt"
	train_path = graph_data_dir / "train.pt"
	val_path = graph_data_dir / "val.pt"
	test_path = graph_data_dir / "test.pt"

	for p in [meta_path, train_path, val_path, test_path]:
		if not p.exists():
			raise FileNotFoundError(f"Missing graph dataset file: {p}")

	metadata = torch.load(meta_path, map_location="cpu")
	train_set = torch.load(train_path, map_location="cpu")
	val_set = torch.load(val_path, map_location="cpu")
	test_set = torch.load(test_path, map_location="cpu")

	def split_summary(name: str, split: dict[str, object]) -> dict[str, object]:
		x = split["x"]
		y = split["y"]
		mask = split["mask"]
		dates = split["trade_dates"]
		if not isinstance(x, torch.Tensor) or not isinstance(mask, torch.Tensor):
			raise TypeError(f"{name} split has invalid tensor types")
		if x.ndim != 3:
			raise ValueError(f"{name} split x shape must be [T,N,F]")

		t, n, f = x.shape
		valid_nodes = int(mask.sum().item())
		total_nodes = int(mask.numel())
		coverage = float(valid_nodes / total_nodes) if total_nodes > 0 else 0.0
		return {
			"steps": int(t),
			"n_nodes": int(n),
			"n_features": int(f),
			"valid_node_labels": valid_nodes,
			"total_node_slots": total_nodes,
			"mask_coverage": coverage,
			"trade_dates_count": int(len(dates)),
			"y_shape": list(y.shape) if isinstance(y, torch.Tensor) else None,
		}

	edge_index = metadata["edge_index"]
	node_codes = metadata["node_codes"]
	feature_cols = metadata["feature_cols"]

	return {
		"graph_data_dir": str(graph_data_dir),
		"edge_count": int(edge_index.shape[1]) if isinstance(edge_index, torch.Tensor) else None,
		"node_count": int(len(node_codes)) if isinstance(node_codes, list) else None,
		"feature_count": int(len(feature_cols)) if isinstance(feature_cols, list) else None,
		"splits": {
			"train": split_summary("train", train_set),
			"val": split_summary("val", val_set),
			"test": split_summary("test", test_set),
		},
	}


def print_brief_report(report: dict[str, object]) -> None:
	source = report["source_dataset"]
	graph = report["graph_dataset"]

	print("=== Source Dataset Summary ===")
	print(f"rows={source['total_rows']} files={source['file_count']}")
	print(f"date_range={source['date_min']} -> {source['date_max']}")
	print(
		"rows_with_any_missing_in_model_inputs="
		f"{source['rows_with_any_missing_in_model_inputs']['count']} "
		f"({source['rows_with_any_missing_in_model_inputs']['ratio']:.6%})"
	)

	missing_items = sorted(
		source["missing_by_col"].items(),
		key=lambda kv: kv[1]["missing_count"],
		reverse=True,
	)
	print("top5_missing_columns:")
	for col, stats in missing_items[:5]:
		print(
			f"  {col}: {stats['missing_count']} ({stats['missing_ratio']:.6%})"
		)

	outlier_items = sorted(
		source["outliers_by_col"].items(),
		key=lambda kv: kv[1]["outlier_count"],
		reverse=True,
	)
	print("top5_outlier_columns_iqr:")
	for col, stats in outlier_items[:5]:
		print(
			f"  {col}: {stats['outlier_count']} "
			f"({stats['outlier_ratio_among_valid']:.6%} of valid)"
		)

	print("=== Graph Dataset Summary ===")
	print(
		f"nodes={graph['node_count']} edges={graph['edge_count']} "
		f"features={graph['feature_count']}"
	)
	for split_name in ["train", "val", "test"]:
		s = graph["splits"][split_name]
		print(
			f"  {split_name}: steps={s['steps']} "
			f"mask_coverage={s['mask_coverage']:.6%}"
		)

	news_focus = source["news_feature_focus"]
	print("=== News Feature Focus ===")
	print("default_fill_ratios:")
	for k, v in news_focus["default_fill_ratios"].items():
		print(f"  {k}: {v:.6%}")
	print("integrity_checks:")
	for k, v in news_focus["integrity_checks"].items():
		print(f"  {k}: {v}")


def main() -> None:
	args = parse_args()
	source_report = analyze_source_dataset(
		dataset_root=args.dataset_root,
		feature_cols=FEATURE_COLS,
		target_col=args.target_col,
	)
	graph_report = analyze_graph_dataset(args.graph_data_dir)

	report = {
		"source_dataset": source_report,
		"graph_dataset": graph_report,
	}

	args.out_json.parent.mkdir(parents=True, exist_ok=True)
	with args.out_json.open("w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)

	print_brief_report(report)
	print(f"report_json={args.out_json}")


if __name__ == "__main__":
	main()
