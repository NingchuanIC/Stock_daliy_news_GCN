from pathlib import Path

import numpy as np
import pandas as pd


feature_cols = [
	"ts_code",
	"trade_date",
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
target_col = "future_ret_5"


def _clean_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
	cleaned = df.copy()
	numeric_cols = [c for c in feature_cols + [target_col] if c not in ["ts_code", "trade_date"]]
	cleaned[numeric_cols] = cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan)
	cleaned = cleaned.dropna(subset=feature_cols + [target_col])
	return cleaned


def _normalize_trade_date(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	if not pd.api.types.is_datetime64_any_dtype(out["trade_date"]):
		out["trade_date"] = pd.to_datetime(
			out["trade_date"].astype(str).str.slice(0, 8),
			format="%Y%m%d",
			errors="coerce",
		)
	return out.dropna(subset=["trade_date"])


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
	out = df.sort_values("trade_date").copy()

	out["ret_1"] = out["close"].pct_change(1)
	out["ret_5"] = out["close"].pct_change(5)
	out["ret_10"] = out["close"].pct_change(10)
	out["ret_20"] = out["close"].pct_change(20)

	out["hl_spread"] = (out["high"] - out["low"]) / out["close"]
	out["co_ret"] = out["close"] / out["open"] - 1

	out["log_vol"] = np.log1p(out["vol"])

	out["ma5"] = out["close"].rolling(5, min_periods=5).mean()
	out["ma10"] = out["close"].rolling(10, min_periods=10).mean()
	out["ma20"] = out["close"].rolling(20, min_periods=20).mean()

	out["ma5_gap"] = out["close"] / out["ma5"] - 1
	out["ma10_gap"] = out["close"] / out["ma10"] - 1
	out["ma20_gap"] = out["close"] / out["ma20"] - 1

	out["volatility_5"] = out["ret_1"].rolling(5, min_periods=5).std()
	out["volatility_20"] = out["ret_1"].rolling(20, min_periods=20).std()

	out[target_col] = out["close"].shift(-5) / out["close"] - 1

	return out


def _build_single_stock(daily_path: Path, basic_path: Path) -> pd.DataFrame:
	df_daily = pd.read_parquet(daily_path)
	df_basic = pd.read_parquet(basic_path)

	daily_need = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
	basic_need = ["ts_code", "trade_date", "turnover_rate", "volume_ratio", "pb"]

	missing_daily = sorted(set(daily_need) - set(df_daily.columns))
	missing_basic = sorted(set(basic_need) - set(df_basic.columns))
	if missing_daily:
		raise ValueError(f"{daily_path.name} missing columns in daily: {missing_daily}")
	if missing_basic:
		raise ValueError(f"{basic_path.name} missing columns in daily_basic: {missing_basic}")

	df_daily = df_daily[daily_need].copy()
	df_basic = df_basic[basic_need].copy()

	df_daily = _normalize_trade_date(df_daily)
	df_basic = _normalize_trade_date(df_basic)

	merged = df_daily.merge(df_basic, on=["ts_code", "trade_date"], how="inner")
	merged = merged.drop_duplicates(subset=["ts_code", "trade_date"])

	for col in [
		"open",
		"high",
		"low",
		"close",
		"vol",
		"turnover_rate",
		"volume_ratio",
		"pb",
	]:
		merged[col] = pd.to_numeric(merged[col], errors="coerce")

	merged = _compute_features(merged)
	merged = merged[feature_cols + [target_col]]
	merged = _clean_feature_rows(merged)
	return merged


def build_all_features(daily_dir: Path, basic_dir: Path) -> pd.DataFrame:
	daily_files = {p.name: p for p in daily_dir.glob("*.parquet")}
	basic_files = {p.name: p for p in basic_dir.glob("*.parquet")}

	common_files = sorted(set(daily_files).intersection(set(basic_files)))
	if not common_files:
		raise FileNotFoundError(
			f"No common parquet files between {daily_dir} and {basic_dir}"
		)

	result_frames: list[pd.DataFrame] = []
	for file_name in common_files:
		one = _build_single_stock(daily_files[file_name], basic_files[file_name])
		if not one.empty:
			result_frames.append(one)

	if not result_frames:
		return pd.DataFrame(columns=feature_cols + [target_col])

	panel = pd.concat(result_frames, ignore_index=True)
	panel = _clean_feature_rows(panel)
	panel = panel.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
	return panel


def write_by_year_month(panel: pd.DataFrame, out_root: Path) -> None:
	out_root.mkdir(parents=True, exist_ok=True)

	panel = panel.copy()
	panel["year"] = panel["trade_date"].dt.year
	panel["month"] = panel["trade_date"].dt.month

	years = range(2019, 2026)
	for year in years:
		year_dir = out_root / str(year)
		year_dir.mkdir(parents=True, exist_ok=True)

		for month in range(1, 13):
			part = panel[(panel["year"] == year) & (panel["month"] == month)].copy()
			part = part.drop(columns=["year", "month"])
			if not part.empty:
				part["trade_date"] = part["trade_date"].dt.strftime("%Y%m%d")
			file_path = year_dir / f"{year}_{month:02d}.parquet"
			part.to_parquet(file_path, index=False)


def main() -> None:
	repo_root = Path(__file__).resolve().parents[1]
	daily_dir = repo_root / "util" / "daliy" / "daily"
	basic_dir = repo_root / "util" / "stock_basic" / "daily_basic"
	out_root = repo_root / "build_feature" / "dataset_by_month"

	panel = build_all_features(daily_dir=daily_dir, basic_dir=basic_dir)
	write_by_year_month(panel=panel, out_root=out_root)

	print(f"Done. Total data points: {len(panel)}")
	print(f"Output directory: {out_root}")


if __name__ == "__main__":
	main()
