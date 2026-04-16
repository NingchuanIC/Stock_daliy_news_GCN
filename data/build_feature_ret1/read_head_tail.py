from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a file and print first N rows and last N rows."
    )
    parser.add_argument("file_path", type=Path, help="Input file path")
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="How many rows to print for head and tail (default: 10)",
    )
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be a positive integer")

    if not args.file_path.exists():
        raise FileNotFoundError(f"File not found: {args.file_path}")

    df = _read_as_dataframe(args.file_path)

    print(f"Loaded: {args.file_path}")
    print(f"Shape: {df.shape}")

    print(f"\n=== Head {args.n} ===")
    print(df.head(args.n))

    print(f"\n=== Tail {args.n} ===")
    print(df.tail(args.n))


if __name__ == "__main__":
    main()
