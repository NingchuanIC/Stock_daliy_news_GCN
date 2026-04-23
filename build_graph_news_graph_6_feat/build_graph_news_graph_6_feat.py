from __future__ import annotations

import argparse
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
	"has_news_7d",
	"news_cnt_7d_log",
	"news_sent_dev_7d",
	"news_sent_std_7d",
	"news_risk_cnt_7d_log",
	"news_freshness",
]

TRAIN_YEARS = {2019, 2020, 2021, 2022, 2023}
VAL_YEARS = {2024}
TEST_YEARS = {2025}


def parse_args() -> argparse.Namespace:
	repo_root = Path(__file__).resolve().parents[1]
	parser = argparse.ArgumentParser(
		description=(
			"Build graph snapshots from monthly parquet features and save train/val/test "
			"datasets for GCN training."
		)
	)
	parser.add_argument(
		"--dataset-root",
		type=Path,
		default=repo_root / "data" / "build_feature_news_graph" / "dataset_by_month",
		help="Root directory containing yearly folders of monthly parquet files.",
	)
	parser.add_argument(
		"--relation-txt",
		type=Path,
		default=repo_root / "build_graph_sector" / "sector_relation_matrix.txt",
		help="Path to relation matrix txt generated from sector mapping.",
	)
	parser.add_argument(
		"--target-col",
		type=str,
		default="future_ret_5",
		help="Target column name in parquet files.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=repo_root / "build_graph_news_graph_6_feat",
		help="Output directory to save metadata/train/val/test torch files.",
	)
	return parser.parse_args()


def load_relation_matrix(relation_txt: Path) -> tuple[list[str], torch.Tensor]:
	if not relation_txt.exists():
		raise FileNotFoundError(f"Relation matrix file not found: {relation_txt}")

	rel_df = pd.read_csv(relation_txt, sep="\t", dtype=str)
	if rel_df.shape[1] < 2:
		raise ValueError("Invalid relation matrix file format.")

	row_codes = rel_df.iloc[:, 0].astype(str).tolist()
	col_codes = [str(c) for c in rel_df.columns[1:]]

	if row_codes != col_codes:
		raise ValueError("Row ts_code order must match column ts_code order.")

	value_df = rel_df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
	if value_df.isna().any().any():
		raise ValueError("Relation matrix contains non-numeric values.")

	adj = value_df.to_numpy(dtype=np.float32)
	if adj.shape[0] != adj.shape[1]:
		raise ValueError("Relation matrix must be square.")

	edge_index_np = np.array(np.nonzero(adj), dtype=np.int64)
	edge_index = torch.tensor(edge_index_np, dtype=torch.long)
	return row_codes, edge_index


def iter_feature_files(dataset_root: Path) -> list[Path]:
	if not dataset_root.exists():
		raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

	files = sorted(dataset_root.glob("*/*.parquet"))
	if not files:
		raise FileNotFoundError(f"No parquet files found under: {dataset_root}")

	return files


def normalize_trade_date_col(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	if not pd.api.types.is_datetime64_any_dtype(out["trade_date"]):
		out["trade_date"] = pd.to_datetime(
			out["trade_date"].astype(str).str.slice(0, 8),
			format="%Y%m%d",
			errors="coerce",
		)
	return out.dropna(subset=["trade_date", "ts_code"])


def build_daily_snapshots(
	dataset_root: Path,
	node_codes: list[str],
	feature_cols: list[str],
	target_col: str,
) -> list[tuple[pd.Timestamp, torch.Tensor, torch.Tensor, torch.Tensor]]:
	code_to_idx = {code: idx for idx, code in enumerate(node_codes)}
	n_nodes = len(node_codes)
	parquet_files = iter_feature_files(dataset_root)

	needed = ["ts_code", "trade_date"] + feature_cols + [target_col]
	snapshots: list[tuple[pd.Timestamp, torch.Tensor, torch.Tensor, torch.Tensor]] = []

	for parquet_path in parquet_files:
		df = pd.read_parquet(parquet_path)
		missing = [c for c in needed if c not in df.columns]
		if missing:
			raise KeyError(f"{parquet_path} missing required columns: {missing}")

		work = normalize_trade_date_col(df[needed])
		for col in feature_cols + [target_col]:
			work[col] = pd.to_numeric(work[col], errors="coerce")

		work["news_sent_std_7d"] = work["news_sent_std_7d"].clip(upper=0.2)
		work["news_freshness"] = np.where(work["has_news_7d"] > 0, work["news_freshness"], 0.0)

		for trade_date, gdf in work.groupby("trade_date", sort=True):
			x_np = np.zeros((n_nodes, len(feature_cols)), dtype=np.float32)
			y_np = np.zeros((n_nodes,), dtype=np.float32)
			mask_np = np.zeros((n_nodes,), dtype=bool)

			for _, row in gdf.iterrows():
				code = str(row["ts_code"])
				idx = code_to_idx.get(code)
				if idx is None:
					continue

				feat = row[feature_cols].to_numpy(dtype=np.float32)
				target = np.float32(row[target_col])

				if np.isnan(feat).any() or np.isnan(target):
					continue

				x_np[idx] = feat
				y_np[idx] = target
				mask_np[idx] = True

			if mask_np.any():
				snapshots.append(
					(
						trade_date,
						torch.tensor(x_np, dtype=torch.float32),
						torch.tensor(y_np, dtype=torch.float32),
						torch.tensor(mask_np, dtype=torch.bool),
					)
				)

	if not snapshots:
		raise ValueError("No valid snapshots were built from monthly parquet files.")

	# Ensure global chronological order across all monthly files.
	snapshots.sort(key=lambda item: item[0])
	return snapshots


def pack_split(
	items: list[tuple[pd.Timestamp, torch.Tensor, torch.Tensor, torch.Tensor]],
	n_nodes: int,
	n_features: int,
) -> dict[str, object]:
	if not items:
		return {
			"x": torch.empty((0, n_nodes, n_features), dtype=torch.float32),
			"y": torch.empty((0, n_nodes), dtype=torch.float32),
			"mask": torch.empty((0, n_nodes), dtype=torch.bool),
			"trade_dates": [],
		}

	x = torch.stack([item[1] for item in items], dim=0)
	y = torch.stack([item[2] for item in items], dim=0)
	mask = torch.stack([item[3] for item in items], dim=0)
	trade_dates = [item[0].strftime("%Y-%m-%d") for item in items]
	return {
		"x": x,
		"y": y,
		"mask": mask,
		"trade_dates": trade_dates,
	}


def split_by_year(
	snapshots: list[tuple[pd.Timestamp, torch.Tensor, torch.Tensor, torch.Tensor]],
	n_nodes: int,
	n_features: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
	train_items: list[tuple[pd.Timestamp, torch.Tensor, torch.Tensor, torch.Tensor]] = []
	val_items: list[tuple[pd.Timestamp, torch.Tensor, torch.Tensor, torch.Tensor]] = []
	test_items: list[tuple[pd.Timestamp, torch.Tensor, torch.Tensor, torch.Tensor]] = []

	for snapshot in snapshots:
		year = snapshot[0].year
		if year in TRAIN_YEARS:
			train_items.append(snapshot)
		elif year in VAL_YEARS:
			val_items.append(snapshot)
		elif year in TEST_YEARS:
			test_items.append(snapshot)

	train_set = pack_split(train_items, n_nodes=n_nodes, n_features=n_features)
	val_set = pack_split(val_items, n_nodes=n_nodes, n_features=n_features)
	test_set = pack_split(test_items, n_nodes=n_nodes, n_features=n_features)
	return train_set, val_set, test_set


def save_datasets(
	output_dir: Path,
	edge_index: torch.Tensor,
	node_codes: list[str],
	feature_cols: list[str],
	target_col: str,
	train_set: dict[str, object],
	val_set: dict[str, object],
	test_set: dict[str, object],
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	metadata = {
		"edge_index": edge_index,
		"node_codes": node_codes,
		"feature_cols": feature_cols,
		"target_col": target_col,
		"train_years": sorted(TRAIN_YEARS),
		"val_years": sorted(VAL_YEARS),
		"test_years": sorted(TEST_YEARS),
	}

	torch.save(metadata, output_dir / "metadata.pt")
	torch.save(train_set, output_dir / "train.pt")
	torch.save(val_set, output_dir / "val.pt")
	torch.save(test_set, output_dir / "test.pt")


def main() -> None:
	args = parse_args()

	node_codes, edge_index = load_relation_matrix(args.relation_txt)
	snapshots = build_daily_snapshots(
		dataset_root=args.dataset_root,
		node_codes=node_codes,
		feature_cols=FEATURE_COLS,
		target_col=args.target_col,
	)

	n_nodes = len(node_codes)
	n_features = len(FEATURE_COLS)
	train_set, val_set, test_set = split_by_year(
		snapshots=snapshots,
		n_nodes=n_nodes,
		n_features=n_features,
	)

	save_datasets(
		output_dir=args.output_dir,
		edge_index=edge_index,
		node_codes=node_codes,
		feature_cols=FEATURE_COLS,
		target_col=args.target_col,
		train_set=train_set,
		val_set=val_set,
		test_set=test_set,
	)

	print(f"Done. Total snapshots: {len(snapshots)}")
	print(f"Train snapshots: {len(train_set['trade_dates'])}")
	print(f"Val snapshots: {len(val_set['trade_dates'])}")
	print(f"Test snapshots: {len(test_set['trade_dates'])}")
	print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
	main()
