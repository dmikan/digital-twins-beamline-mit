"""
Compare the last column (V) across multiple basis_electrode_*.csv files.

Usage:
    python compare_basis_files.py /path/to/folder
    python compare_basis_files.py /path/to/folder --pattern "basis_electrode_*.csv"

Checks:
    1. Whether all files have identical V columns (row-by-row).
    2. Pairwise comparison summary (which files match which).
    3. Basic stats per file (min/max/mean/nonzero count) as a sanity check.
"""

import sys
import glob
import argparse
import csv
import os


def load_v_column(filepath):
    """Read the last column (assumed to be V) from a CSV file with a header row."""
    values = []
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            if not row:
                continue
            values.append(float(row[-1]))
    return values


def stats(values):
    n = len(values)
    nonzero = sum(1 for v in values if v != 0.0)
    vmin = min(values) if values else float("nan")
    vmax = max(values) if values else float("nan")
    vmean = sum(values) / n if n else float("nan")
    return {
        "count": n,
        "nonzero": nonzero,
        "min": vmin,
        "max": vmax,
        "mean": vmean,
    }


def columns_equal(a, b, tol=1e-9):
    if len(a) != len(b):
        return False, "different row counts"
    for i, (x, y) in enumerate(zip(a, b)):
        if abs(x - y) > tol:
            return False, f"first difference at row {i}: {x} vs {y}"
    return True, "identical"


def main():
    parser = argparse.ArgumentParser(description="Compare V columns across basis_electrode CSV files.")
    parser.add_argument("folder", nargs="?", default=".", help="Folder containing the CSV files (default: current dir)")
    parser.add_argument("--pattern", default="basis_electrode_*.csv", help="Glob pattern to match files")
    parser.add_argument("--tol", type=float, default=1e-9, help="Numerical tolerance for equality check")
    args = parser.parse_args()

    search_path = os.path.join(args.folder, args.pattern)
    files = sorted(glob.glob(search_path))

    if not files:
        print(f"No files found matching: {search_path}")
        sys.exit(1)

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  - {f}")
    print()

    # Load V columns
    data = {}
    for f in files:
        print(f"Loading {f} ...")
        data[f] = load_v_column(f)

    print()
    print("=== Per-file stats ===")
    for f in files:
        s = stats(data[f])
        print(f"{os.path.basename(f)}: count={s['count']} nonzero={s['nonzero']} "
              f"min={s['min']:.6f} max={s['max']:.6f} mean={s['mean']:.6f}")

    print()
    print("=== Pairwise comparison (vs first file) ===")
    reference_file = files[0]
    reference_values = data[reference_file]
    all_identical = True
    for f in files[1:]:
        equal, detail = columns_equal(reference_values, data[f], tol=args.tol)
        status = "IDENTICAL" if equal else "DIFFERENT"
        if not equal:
            all_identical = False
        print(f"{os.path.basename(reference_file)} vs {os.path.basename(f)}: {status} ({detail})")

    print()
    if all_identical:
        print("RESULT: All files have identical V columns.")
        print("        This likely means the basis extraction is NOT isolating electrodes correctly.")
    else:
        print("RESULT: V columns differ across files.")
        print("        This is what you want -- each electrode is producing a distinct basis field.")


if __name__ == "__main__":
    main()