from pathlib import Path
import json

import numpy as np
import pandas as pd


NEWS_SENTIMENT_DEFAULT = 0.732936
NEWS_POS_THRESHOLD = 0.8
NEWS_NEG_THRESHOLD = 0.6
NEWS_RISK_THRESHOLD = 0.4

news_feature_cols = [
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


price_feature_cols = [
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


feature_cols = [
	*price_feature_cols,
	*news_feature_cols,
]
target_col = "future_ret_5"


def _clean_feature_rows(df: pd.DataFrame, required_feature_cols: list[str] | None = None) -> pd.DataFrame:
	cleaned = df.copy()
	if required_feature_cols is None:
		required_feature_cols = feature_cols
	numeric_cols = [
		c
		for c in required_feature_cols + [target_col]
		if c not in ["ts_code", "trade_date"] and c in cleaned.columns
	]
	cleaned[numeric_cols] = cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan)
	required_existing = [c for c in required_feature_cols + [target_col] if c in cleaned.columns]
	cleaned = cleaned.dropna(subset=required_existing)
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


def _ts_code_to_6_digits(series: pd.Series) -> pd.Series:
	return series.astype(str).str.extract(r"(\d{6})", expand=False)


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


def _iter_news_files(news_root: Path):
	for year_dir in sorted(news_root.glob("*")):
		if not year_dir.is_dir():
			continue
		for month_dir in sorted(year_dir.glob("*")):
			if not month_dir.is_dir():
				continue
			for news_file in sorted(month_dir.glob("*.txt")):
				yield news_file


def _load_news_events(news_root: Path) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for file_path in _iter_news_files(news_root):
		with file_path.open("r", encoding="utf-8") as f:
			for line in f:
				text = line.strip()
				if not text:
					continue
				try:
					obj = json.loads(text)
				except json.JSONDecodeError:
					continue

				ts_code = obj.get("ts_code")
				date_str = obj.get("date")
				score = obj.get("sentiment_score")
				if ts_code is None or date_str is None or score is None:
					continue

				rows.append(
					{
						"ts_code6": str(ts_code).zfill(6)[:6],
						"trade_date": str(date_str)[:10],
						"sentiment_score": score,
					}
				)

	if not rows:
		return pd.DataFrame(columns=["ts_code6", "trade_date", "sentiment_score"])

	news_df = pd.DataFrame(rows)
	news_df["trade_date"] = pd.to_datetime(news_df["trade_date"], format="%Y-%m-%d", errors="coerce")
	news_df["sentiment_score"] = pd.to_numeric(news_df["sentiment_score"], errors="coerce")
	news_df = news_df.dropna(subset=["ts_code6", "trade_date", "sentiment_score"])
	return news_df


def _compute_days_since_last_news(trade_dates: pd.Series, has_news: pd.Series) -> pd.Series:
	date_index = pd.DatetimeIndex(trade_dates)
	last_news = pd.Series(date_index.where(has_news.to_numpy()), index=date_index).ffill()
	days = (date_index - pd.DatetimeIndex(last_news)).days
	return pd.Series(days, index=trade_dates.index).fillna(9999).astype(np.int32)


def _rolling_news_features(group: pd.DataFrame) -> pd.DataFrame:
	g = group.sort_values("trade_date").copy()
	if "ts_code6" not in g.columns:
		g["ts_code6"] = str(group.name)

	g["news_cnt_3d"] = g["news_cnt_1d"].rolling(3, min_periods=1).sum()
	g["news_cnt_7d"] = g["news_cnt_1d"].rolling(7, min_periods=1).sum()

	g["news_sent_sum_3d"] = g["news_sent_sum_1d"].rolling(3, min_periods=1).sum()
	g["news_sent_sum_7d"] = g["news_sent_sum_1d"].rolling(7, min_periods=1).sum()

	g["news_sent_mean_1d"] = np.where(
		g["news_cnt_1d"] > 0,
		g["news_sent_sum_1d"] / g["news_cnt_1d"],
		np.nan,
	)
	g["news_sent_mean_3d"] = np.where(
		g["news_cnt_3d"] > 0,
		g["news_sent_sum_3d"] / g["news_cnt_3d"],
		np.nan,
	)
	g["news_sent_mean_7d"] = np.where(
		g["news_cnt_7d"] > 0,
		g["news_sent_sum_7d"] / g["news_cnt_7d"],
		np.nan,
	)

	# Max/min/std ignore no-news days by using NaN for empty days.
	g["news_sent_std_7d"] = g["news_sent_mean_1d"].rolling(7, min_periods=1).std(ddof=0)
	g["news_sent_max_3d"] = g["news_sent_mean_1d"].rolling(3, min_periods=1).max()
	g["news_sent_min_3d"] = g["news_sent_mean_1d"].rolling(3, min_periods=1).min()

	g["news_pos_cnt_3d"] = g["news_pos_cnt_1d"].rolling(3, min_periods=1).sum()
	g["news_neg_cnt_3d"] = g["news_neg_cnt_1d"].rolling(3, min_periods=1).sum()
	g["news_risk_cnt_7d"] = g["news_risk_cnt_1d"].rolling(7, min_periods=1).sum()

	has_news = g["news_cnt_1d"] > 0
	g["days_since_last_news"] = _compute_days_since_last_news(g["trade_date"], has_news)

	return g[["ts_code6", "trade_date"] + news_feature_cols]


def _build_news_features(news_root: Path, stock_panel: pd.DataFrame) -> pd.DataFrame:
	stock_dates = stock_panel[["ts_code", "trade_date"]].copy()
	stock_dates["ts_code6"] = _ts_code_to_6_digits(stock_dates["ts_code"])
	stock_dates = stock_dates.dropna(subset=["ts_code6", "trade_date"])

	all_codes = sorted(stock_dates["ts_code6"].unique())
	start_date = stock_dates["trade_date"].min()
	end_date = stock_dates["trade_date"].max()
	all_dates = pd.date_range(start=start_date, end=end_date, freq="D")

	news_events = _load_news_events(news_root)
	if news_events.empty:
		base_grid = pd.MultiIndex.from_product(
			[all_codes, all_dates], names=["ts_code6", "trade_date"]
		).to_frame(index=False)
		base_grid["news_cnt_1d"] = 0
		base_grid["news_sent_sum_1d"] = 0.0
		base_grid["news_pos_cnt_1d"] = 0
		base_grid["news_neg_cnt_1d"] = 0
		base_grid["news_risk_cnt_1d"] = 0
	else:
		news_events = news_events[news_events["ts_code6"].isin(all_codes)]
		news_events = news_events[
			(news_events["trade_date"] >= start_date)
			& (news_events["trade_date"] <= end_date)
		]

		news_events["is_pos"] = (news_events["sentiment_score"] >= NEWS_POS_THRESHOLD).astype(np.int16)
		news_events["is_neg"] = (news_events["sentiment_score"] < NEWS_NEG_THRESHOLD).astype(np.int16)
		news_events["is_risk"] = (news_events["sentiment_score"] < NEWS_RISK_THRESHOLD).astype(np.int16)

		base_daily = (
			news_events.groupby(["ts_code6", "trade_date"], as_index=False)
			.agg(
				news_cnt_1d=("sentiment_score", "size"),
				news_sent_sum_1d=("sentiment_score", "sum"),
				news_pos_cnt_1d=("is_pos", "sum"),
				news_neg_cnt_1d=("is_neg", "sum"),
				news_risk_cnt_1d=("is_risk", "sum"),
			)
		)

		full_grid = pd.MultiIndex.from_product(
			[all_codes, all_dates], names=["ts_code6", "trade_date"]
		).to_frame(index=False)
		base_grid = full_grid.merge(base_daily, on=["ts_code6", "trade_date"], how="left")

	base_grid["news_cnt_1d"] = base_grid["news_cnt_1d"].fillna(0).astype(np.int32)
	base_grid["news_sent_sum_1d"] = base_grid["news_sent_sum_1d"].fillna(0.0)
	base_grid["news_pos_cnt_1d"] = base_grid["news_pos_cnt_1d"].fillna(0).astype(np.int32)
	base_grid["news_neg_cnt_1d"] = base_grid["news_neg_cnt_1d"].fillna(0).astype(np.int32)
	base_grid["news_risk_cnt_1d"] = base_grid["news_risk_cnt_1d"].fillna(0).astype(np.int32)

	rolled = (
		base_grid.groupby("ts_code6", group_keys=False)
		.apply(_rolling_news_features)
		.reset_index(drop=True)
	)

	stock_dates = stock_dates[["ts_code6", "trade_date"]].drop_duplicates()
	rolled = stock_dates.merge(rolled, on=["ts_code6", "trade_date"], how="left")
	rolled = rolled.sort_values(["ts_code6", "trade_date"]).reset_index(drop=True)

	# Trading assumption: use day t news to predict returns from day t+1 onward.
	rolled[news_feature_cols] = rolled.groupby("ts_code6", group_keys=False)[news_feature_cols].shift(1)

	for col in news_feature_cols:
		if col.startswith("news_cnt") or col.endswith("_cnt_3d") or col.endswith("_cnt_7d"):
			rolled[col] = rolled[col].fillna(0)

	rolled["news_sent_mean_1d"] = rolled["news_sent_mean_1d"].fillna(NEWS_SENTIMENT_DEFAULT)
	rolled["news_sent_mean_3d"] = rolled["news_sent_mean_3d"].fillna(NEWS_SENTIMENT_DEFAULT)
	rolled["news_sent_mean_7d"] = rolled["news_sent_mean_7d"].fillna(NEWS_SENTIMENT_DEFAULT)
	rolled["news_sent_std_7d"] = rolled["news_sent_std_7d"].fillna(0.0)
	rolled["news_sent_max_3d"] = rolled["news_sent_max_3d"].fillna(NEWS_SENTIMENT_DEFAULT)
	rolled["news_sent_min_3d"] = rolled["news_sent_min_3d"].fillna(NEWS_SENTIMENT_DEFAULT)
	rolled["days_since_last_news"] = rolled["days_since_last_news"].fillna(9999).astype(np.int32)

	count_cols = [
		"news_cnt_1d",
		"news_cnt_3d",
		"news_cnt_7d",
		"news_pos_cnt_3d",
		"news_neg_cnt_3d",
		"news_risk_cnt_7d",
	]
	rolled[count_cols] = rolled[count_cols].round().astype(np.int32)

	return rolled


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
	merged = merged[price_feature_cols + [target_col]]
	merged = _clean_feature_rows(merged, required_feature_cols=price_feature_cols)
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
	panel["ts_code6"] = _ts_code_to_6_digits(panel["ts_code"])

	repo_root = Path(__file__).resolve().parents[1]
	news_root = repo_root / "util" / "news" / "news_score_500"
	news_panel = _build_news_features(news_root=news_root, stock_panel=panel)
	panel = panel.merge(news_panel, on=["ts_code6", "trade_date"], how="left")
	panel = panel.drop(columns=["ts_code6"])

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
	out_root = repo_root / "build_feature_news" / "dataset_by_month"

	panel = build_all_features(daily_dir=daily_dir, basic_dir=basic_dir)
	write_by_year_month(panel=panel, out_root=out_root)

	print(f"Done. Total data points: {len(panel)}")
	print(f"Output directory: {out_root}")


if __name__ == "__main__":
	main()
