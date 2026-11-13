#!/usr/bin/env python3
"""Write a small CSV grid for cuda-2d-stencil (comma-separated values, one row per line)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample 2D stencil input CSV")
    parser.add_argument("-o", "--output", default="sample_grid.csv", help="Output file path")
    parser.add_argument("--width", type=int, default=8, help="Number of columns")
    parser.add_argument("--height", type=int, default=8, help="Number of rows")
    parser.add_argument(
        "--pattern",
        choices=("checker", "gradient", "center"),
        default="checker",
        help="Initial field pattern",
    )
    args = parser.parse_args()

    rows: list[list[float]] = []
    for r in range(args.height):
        row: list[float] = []
        for c in range(args.width):
            if args.pattern == "checker":
                v = 1.0 if (r + c) % 2 == 0 else 0.0
            elif args.pattern == "gradient":
                v = (r + c) / max(args.width + args.height - 2, 1)
            else:
                mid_r, mid_c = args.height // 2, args.width // 2
                v = 1.0 if abs(r - mid_r) + abs(c - mid_c) <= 1 else 0.0
            row.append(v)
        rows.append(row)

    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(f"{x:.3f}" for x in row))
            f.write("\n")

    print(f"Wrote {args.height}x{args.width} grid to {args.output}")


if __name__ == "__main__":
    main()
