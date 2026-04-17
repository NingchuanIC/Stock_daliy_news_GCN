import argparse
from pathlib import Path
from typing import List, Tuple


def load_stock_sector_map(file_path: Path) -> List[Tuple[str, str]]:
    """Load (ts_code, sector) pairs from a tab-separated text file."""
    rows: List[Tuple[str, str]] = []

    with file_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if len(header) < 4:
            raise ValueError("Input file must contain at least 4 tab-separated columns.")

        try:
            ts_code_idx = header.index("ts_code")
            sector_idx = header.index("sector")
        except ValueError as exc:
            raise ValueError("Header must include 'ts_code' and 'sector' columns.") from exc

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) <= max(ts_code_idx, sector_idx):
                continue
            ts_code = parts[ts_code_idx].strip()
            sector = parts[sector_idx].strip()
            if ts_code and sector:
                rows.append((ts_code, sector))

    if not rows:
        raise ValueError("No valid data rows found in input file.")

    return rows


def build_relation_matrix(data: List[Tuple[str, str]]) -> Tuple[List[str], List[List[int]]]:
    """Build adjacency matrix: 1 if same sector else 0."""
    ts_codes = [item[0] for item in data]
    sectors = [item[1] for item in data]
    n = len(ts_codes)

    matrix: List[List[int]] = []
    for i in range(n):
        row: List[int] = []
        for j in range(n):
            row.append(1 if sectors[i] == sectors[j] else 0)
        matrix.append(row)

    return ts_codes, matrix


def save_matrix_txt(output_path: Path, ts_codes: List[str], matrix: List[List[int]]) -> None:
    """Save an (N+1) x (N+1) matrix where first row/column are ts_code labels."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("ts_code\t" + "\t".join(ts_codes) + "\n")
        for i, code in enumerate(ts_codes):
            values = "\t".join(str(v) for v in matrix[i])
            f.write(f"{code}\t{values}\n")


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Build stock relation matrix by sector (1=same sector, 0=otherwise)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("zz500_stock_sector_map.txt"),
        help="Path to stock-sector mapping file (tab-separated).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sector_relation_matrix.txt"),
        help="Output txt path for the (N+1) x (N+1) relation matrix.",
    )
    args = parser.parse_args()

    # Resolve relative paths against script directory, not current shell cwd.
    input_path = args.input if args.input.is_absolute() else script_dir / args.input
    output_path = args.output if args.output.is_absolute() else script_dir / args.output

    data = load_stock_sector_map(input_path)
    ts_codes, matrix = build_relation_matrix(data)
    save_matrix_txt(output_path, ts_codes, matrix)

    print(f"Done. Stocks: {len(ts_codes)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
