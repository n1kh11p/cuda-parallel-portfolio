# CUDA 2D five-point stencil

Iterative **five-point stencil** on a 2D grid: each interior cell becomes the average of itself and its four axis-aligned neighbors (left, right, up, down). One timestep applies

\[
\text{next}[r,c] = 0.2 \times (\text{center} + \text{left} + \text{right} + \text{up} + \text{down})
\]

with values read from a **padded** copy of the previous grid. Edge padding uses periodic boundary handling via `padded_matrix_copy` (course assignment design).

Completed kernels come from **CMSC416 Assignment 4**; packaged here for portfolio and local builds.

## What the code demonstrates

- **`compute_on_gpu`**: strided 2D thread loops over `X_limit × Y_limit` interior cells.
- **`padded_matrix_copy`**: copies the live grid into a ghost-cell buffer and mirrors edges for neighbors at boundaries.
- **Host loop**: for each iteration, pad-copy then stencil; timing with CUDA events around the whole iteration loop.

This is a building block for diffusion-like updates, Jacobi-style iterations, and finite-difference patterns—not Game of Life (despite an outdated comment in `main`).

## Requirements

- NVIDIA GPU and CUDA toolkit (`nvcc`)

## Build

```bash
cd cuda-2d-stencil
make
```

Override architecture if needed:

```bash
CUDA_ARCH=80 make clean && make
```

## Input format

Comma-separated values, **one row per line**, `Y_limit` rows and `X_limit` columns. A small example ships as `sample_grid.csv` (8×8 checker). Generate others from the repo root:

```bash
python3 scripts/generate_stencil_input.py -o cuda-2d-stencil/my_grid.csv --width 16 --height 16
```

## Run

```bash
./2d-stencil sample_grid.csv 100 8 8 1 1 out.csv
```

Arguments (8 total including the program name):

| Position | Meaning |
|----------|---------|
| `argv[1]` | Input CSV path |
| `argv[2]` | Number of stencil iterations |
| `argv[3]`, `argv[4]` | `X_limit`, `Y_limit` (must match file) |
| `argv[5]`, `argv[6]` | `gridSizeX`, `gridSizeY` for kernel launch |
| `argv[7]` | Output CSV path |

Stdout reports `Runtime: … s` for the timed iteration loop (CUDA events). Run on your GPU to obtain numbers; none are quoted in this README.

## How to measure

1. Fix grid size, iteration count, and launch grid; run several times.
2. Record `Runtime` from the program, or add per-iteration events if you extend the code.
3. Validate correctness on a tiny grid by comparing a few steps against a serial Python/NumPy reference.
4. Document GPU model and driver/CUDA versions when sharing results.

## Layout

| File | Role |
|------|------|
| `2d-stencil.cu` | Kernels, I/O, `main` |
| `sample_grid.csv` | Tiny starter input |
| `Makefile` | Single-file `nvcc` build |
