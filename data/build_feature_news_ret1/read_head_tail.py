from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path(
    r"D:\imperial_homework\third_year\new_personal_project\build_feature_news\dataset_by_month\2019\2019_01.parquet"
)


def _read_as_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(file_path)

    if suffix in {".csv", ".txt"}:
        # For txt/csv, default to comma-separated; if it fails, fall back to whitespace.
        try:
            return pd.read_csv(file_path)
        except Exception:
            return pd.read_csv(file_path, sep=r"\s+", engine="python")

    raise ValueError(f"Unsupported file type: {suffix}. Use .parquet/.csv/.txt")


def _resolve_input_path(input_path: Path, year: int | None, month: int | None) -> Path:
    if input_path.is_file():
        return input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    if year is not None or month is not None:
        if year is None or month is None:
            raise ValueError("Please provide both --year and --month together")
        if month < 1 or month > 12:
            raise ValueError("--month must be in 1..12")
        candidate = input_path / str(year) / f"{year}_{month:02d}.parquet"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Parquet not found: {candidate}")

    files = sorted(input_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {input_path}")
    return files[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a file and print first N rows and last N rows."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input file path, or dataset directory (e.g., dataset_by_month)",
    )
    parser.add_argument("--year", type=int, default=None, help="Year for dataset directory")
    parser.add_argument("--month", type=int, default=None, help="Month for dataset directory")
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="How many rows to print for head and tail (default: 10)",
    )
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be a positive integer")

    target_path = _resolve_input_path(args.input_path, args.year, args.month)
    df = _read_as_dataframe(target_path)

    print(f"Loaded: {target_path}")
    print(f"Shape: {df.shape}")

    print(f"\n=== Head {args.n} ===")
    print(df.head(args.n))

    print(f"\n=== Tail {args.n} ===")
    print(df.tail(args.n))


if __name__ == "__main__":
    main()
