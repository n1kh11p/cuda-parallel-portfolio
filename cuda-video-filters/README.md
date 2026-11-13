# CUDA video filters (2D convolution)

GPU-accelerated **spatial filtering** on video frames: each output pixel is a weighted sum of a small neighborhood in the input (a 2D convolution). This is classical image processing—not neural network training or inference.

Completed kernels and host orchestration come from **CMSC416 Assignment 4** (University of Maryland); this repo packages them for local builds and portfolio use.

## What the code demonstrates

- **`convolveGPU`**: 2D strided grid mapping so one thread block can cover frames larger than the launch grid; per-channel (RGB) dot products against a fixed kernel; border pixels skipped so the kernel stays fully inside the frame.
- **`convolveFrames`**: launches one convolution per frame in a batch, round-robin across **CUDA streams** to overlap independent frame work.
- **`convolveCPU`**: reference implementation for correctness checks (not used in the default driver path).
- **Driver** (`driver.cpp`): OpenCV reads/writes MP4, batches frames on the device, and reports elapsed GPU time via CUDA events.

Built-in kernels: `identity` (3×3), `edge`, `sharpen`, `blur` (5×5).

### Streams (why they matter)

Each stream is an ordered queue of GPU work. Assigning consecutive frames to different streams (`i % numStreams`) lets the device start the next frame’s kernel while earlier frames may still be running, which improves throughput when batches contain multiple frames. Synchronization (`cudaDeviceSynchronize` inside `convolveFrames`) bounds the timed region per batch.

## Requirements

- NVIDIA GPU with a matching CUDA toolkit (`nvcc`)
- OpenCV 4.x with `videoio` (for the host driver)
- C++11 compiler for linking (`g++`)

## Build

```bash
cd cuda-video-filters
make
```

**Overrides** (if `pkg-config` does not find OpenCV):

```bash
export CUDA_HOME=/usr/local/cuda
export CUDA_ARCH=80          # or: native
export OPENCV_PREFIX=/path/to/opencv
make clean && make
```

On Debian/Ubuntu, `libopencv-dev` and `pkg-config` often satisfy OpenCV. Set `CUDA_ARCH` to your GPU’s compute capability (see [NVIDIA docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)) or use `native` when `nvcc` supports it.

## Run

Provide your own input video (sample MP4s are not committed—see root `.gitignore`).

```bash
./video-effect input.mp4 output.mp4 sharpen 120 68
```

Arguments:

| Arg | Meaning |
|-----|---------|
| `input.mp4` / `output.mp4` | Source and destination video |
| `kernel_name` | `identity`, `edge`, `sharpen`, or `blur` |
| `grid_size_x`, `grid_size_y` | CUDA grid dimensions passed to `convolveGPU` (assignment-style tuning knob) |

Optional: dump one processed frame as CSV for debugging:

```bash
./video-effect input.mp4 output.mp4 edge 120 68 0 frame0.csv
```

The program prints **total kernel time** (seconds) and **frames/s** derived from batching; treat these as **on-your-machine** measurements, not published benchmarks.

## How to measure performance (no invented numbers)

1. Build in release mode (`-O2`, default in the Makefile).
2. Use a fixed clip, kernel, and grid size; run multiple times and note variance.
3. Read `Total time` and `Frames processed per second` from stdout, or wrap `convolveFrames` with your own `cudaEvent` timing.
4. Use `nvidia-smi` to confirm GPU model, clock behavior, and that no other jobs contend for the device.
5. Compare against CPU with `convolveCPU` on a single frame if you add a small test harness—this repo keeps the course driver unchanged.

## Layout

| File | Role |
|------|------|
| `video-effect.cu` | Kernels + `convolveFrames` |
| `driver.cpp` | Video I/O, streams, CLI |
| `Makefile` | Portable `nvcc` + OpenCV link |
