"""Calculate a pixel-wise difference between two NumPy arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def calculate_difference(
    first_path: Path,
    second_path: Path,
    output_path: Path,
    *,
    absolute: bool = False,
) -> np.ndarray:
    """Save and return ``first - second`` for two arrays with identical shapes."""
    first = np.load(first_path, allow_pickle=False)
    second = np.load(second_path, allow_pickle=False)

    if first.shape != second.shape:
        raise ValueError(
            "Input shapes must match: "
            f"{first_path} has {first.shape}, but {second_path} has {second.shape}."
        )
    if not np.issubdtype(first.dtype, np.number) or not np.issubdtype(
        second.dtype, np.number
    ):
        raise TypeError(
            f"Inputs must contain numeric values, got {first.dtype} and {second.dtype}."
        )

    result_dtype = np.result_type(first.dtype, second.dtype, np.float32)
    difference = np.subtract(first, second, dtype=result_dtype)
    if absolute:
        difference = np.abs(difference)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, difference, allow_pickle=False)
    return difference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the pixel-wise difference between two same-shaped NPY files."
    )
    parser.add_argument("first", type=Path, help="First NPY file (minuend).")
    parser.add_argument("second", type=Path, help="Second NPY file (subtrahend).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("diff.npy"),
        help="Output NPY path (default: diff.npy).",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Save abs(first - second) instead of the signed difference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    difference = calculate_difference(
        args.first,
        args.second,
        args.output,
        absolute=args.absolute,
    )
    print(
        f"Saved {args.output} | shape={difference.shape}, dtype={difference.dtype}, "
        f"min={difference.min()}, max={difference.max()}"
    )


if __name__ == "__main__":
    main()
