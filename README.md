# CUDA parallel portfolio

Small, finished CUDA projects from parallel programming coursework, cleaned up for cloning and building on a Linux machine with an NVIDIA GPU.

**Origin:** Kernels and drivers are from [CMSC416](https://www.cs.umd.edu/class/) *Introduction to Parallel Computing*, Assignment 4 (University of Maryland). They were completed as course work; this repository is an honest portfolio packaging—not a from-scratch research codebase. Stubs under `~/CMSC416/assignment-4` were incomplete; the implementations here match the finished copies in personal assignment archives.

## Projects

| Directory | Topic | Highlights |
|-----------|--------|------------|
| [`cuda-video-filters/`](cuda-video-filters/) | 2D convolution on video frames | Strided `convolveGPU`, multi-stream frame batches, OpenCV I/O |
| [`cuda-2d-stencil/`](cuda-2d-stencil/) | Five-point stencil iterations | Padded periodic boundaries, Jacobi-style GPU updates |

Neither project trains or runs neural networks; they focus on **kernel design**, **launch configuration**, and **basic GPU orchestration** (streams, events, device memory).

## Hardware and environment

- **GPU:** Any CUDA-capable NVIDIA GPU whose architecture you pass to the Makefiles (`CUDA_ARCH` or `native`).
- **OS:** Developed and graded on Linux (UMD clusters). macOS does not run CUDA locally; use a Linux box, lab machine, or cloud GPU instance.
- **Software:** CUDA toolkit (`nvcc`), `g++`, and (for video filters only) OpenCV 4 with video I/O.

Check your device:

```bash
nvidia-smi
nvcc --version
```

## Quick build

```bash
# Video convolution + streams
cd cuda-video-filters && make && cd ..

# 2D stencil
cd cuda-2d-stencil && make && cd ..
```

See each project README for run examples, CLI arguments, and **how to measure** performance on your own hardware. This repo does not ship benchmark tables.

## Repository layout

```
cuda-parallel-portfolio/
├── README.md                 # This file
├── .gitignore
├── scripts/
│   └── generate_stencil_input.py
├── cuda-video-filters/
│   ├── video-effect.cu
│   ├── driver.cpp
│   ├── Makefile
│   └── README.md
└── cuda-2d-stencil/
    ├── 2d-stencil.cu
    ├── sample_grid.csv
    ├── Makefile
    └── README.md
```

## Sample media

Input videos for `cuda-video-filters` are **not** tracked in git (size). Use any MP4 readable by OpenCV, or keep clips outside the repo. See `.gitignore`.

## License / attribution

Course materials are © University of Maryland; student implementations are presented here for portfolio purposes. Do not submit this repo as coursework if your institution forbids reuse of prior assignments.
