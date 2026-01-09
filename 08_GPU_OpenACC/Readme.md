# High-Performance OpenACC Laplace Solver

**Course:** Parallel Programming (GPU Programming)  
**Group:** 22  
**Target Hardware:** NVIDIA RTX 2070 (Partition: `cuda-ext.q`)  
**Status:** ✅ Optimized (2.96x Speedup over Static Baseline)

---

## 📌 Project Overview

This repository contains the source code, benchmark scripts, and performance analysis tools for a GPU-accelerated Laplace equation solver using OpenACC.

The primary objective of this assignment was to refactor a static, compile-time C code (`laplace_baseline.c`) into a flexible, dynamic memory implementation (`laplace_opt.c`) that supports runtime mesh sizing while outperforming the static version.

### Key Achievements

* **Runtime Flexibility:** Replaced static 2D arrays (`float A[N][M]`) with flattened 1D dynamic arrays (`malloc`), allowing mesh sizes to be defined via command-line arguments.
* **Performance:** Achieved a **2.96x speedup** (4.14s vs 12.26s) on a 4096 x 4096 grid.
* **Memory Saturation:** Reached an effective bandwidth of **~323 GB/s**, effectively utilizing 72% of the RTX 2070's theoretical peak (448 GB/s). We have hit the physical "Memory Wall."
* **Latency Hiding:** Implemented asynchronous execution with periodic error checking to keep the GPU pipeline continuously saturated.

---

## 📂 Repository Structure

The project is organized to separate source code, execution logic, and generated data.

```text
.
├── src/
│   ├── laplace_baseline.c    # Original static version (Reference implementation)
│   └── laplace_opt.c         # Final optimized version (V4: Dynamic, Collapse, Async)
│
├── scripts/
│   ├── run_full.job          # MAIN SCRIPT: Compiles, benchmarks, and profiles both versions
│   ├── run_test.job          # Fast functional test (small grid) to check for errors
│   ├── run_plotting.job      # Generates the Performance Bar Chart
│   ├── run_timeline_plot.job # Generates the Gantt Chart (Timeline visualization)
│   ├── plot_results.py       # Python logic for bar charts (Robust CSV parsing)
│   └── plot_timeline.py      # Python logic for Nsight Systems trace analysis
│
└── results/                  # Auto-generated artifacts (do not commit large files here)
    ├── log/                  # SLURM output logs (stdout)
    ├── err/                  # SLURM error logs (stderr)
    ├── csv/                  # Timing data (e.g., final_comparison.csv)
    ├── plots/                # Generated PNG visualizations for the report
    └── profiling/            # Nsight Systems binary files (.qdrep, .sqlite)

```

---

## 🚀 Setup & Usage Guide

### 1. Prerequisites

You must be on the cluster login node. Load the NVIDIA HPC SDK:

```bash
module purge
module load nvhpc/21.2

```

### 2. Running the Full Benchmark

To compile the code, run the performance comparison (4096^2, 10k iters), and generate profiling data:

```bash
sbatch scripts/run_full.job

```

* **Monitor:** `squeue`
* **Output:** Check `results/log/lap_full_*.log` for the textual timing results.
* **Artifacts:** The binary profiling files (`.qdrep`) will be automatically moved to `results/profiling/`.

### 3. Generating Visualizations (For the Report)

Once the benchmark job finishes, generate the plots.

**A. Performance Bar Chart:** This creates a chart comparing Baseline vs. V1, V2, V3, and V4.

```bash
sbatch scripts/run_plotting.job

```

> **Output:** `results/plots/performance_comparison.png`

**B. Execution Timeline (Gantt Chart):** This creates a visual comparison of kernel density (Sparse vs. Dense).

```bash
sbatch scripts/run_timeline_plot.job

```

> **Output:** `results/plots/timeline_comparison.png`

---

## ⚡ Optimization Strategy

We applied four specific techniques to transform the code from "Slow & Static" to "Fast & Dynamic".

### 1. Flattened Dynamic Memory (Coalescing)

Instead of `float A[n][m]` (which is only guaranteed contiguous in specific static cases), we used:

```c
float * restrict A = (float*) malloc(n * m * sizeof(float));
// Access: A[j*m + i]

```

* **Why:** Flattening guarantees that rows are contiguous in physical memory. This allows the GPU to perform **Coalesced Memory Access**, where a warp (32 threads) reads a continuous 128-byte chunk in a single transaction.
* **Restrict:** The `restrict` keyword promises the compiler that `A` and `Anew` do not overlap, enabling aggressive vectorization.

### 2. Pointer Swapping ("Ping-Pong")

The baseline used a dedicated kernel to copy `Anew` → `A` at the end of every step. This is an O(N) operation purely for data movement.

* **Solution:** We swap pointers on the CPU: `float *temp = input; input = output; output = temp;`
* **Impact:** Reduced global memory traffic by ~33% per iteration.

### 3. Periodic Error Checking (Latency Hiding)

Checking convergence (reduction) every iteration forces a CPU-GPU synchronization barrier.

* **Solution:** We check the error only every 100 iterations (`if (iter % 100 == 0)`).
* **Impact:** The GPU executes 99 iterations purely asynchronously (`async(1)`), queuing kernel after kernel without waiting for the host.

### 4. Occupancy Optimization (collapse)

We used `#pragma acc parallel loop collapse(2)`.

* **Why:** This flattens the loop into a single dimension of 16,000,000 threads. It allows the NVIDIA scheduler to maximize **Occupancy** (active warps). If one warp stalls on memory, the scheduler instantly switches to another, hiding the memory latency.

---

## 🚧 Hurdles & Failed Attempts

It is important to understand what *didn't* work to avoid regressing in the future.

### The "Warp-Alignment" Failure (Version 3)

We attempted to manually tune the cache usage by using:

```c
#pragma acc parallel loop tile(4, 32) vector_length(32)

```

* **The Logic:** We wanted the inner tile dimension (32) to match the GPU warp size (32) exactly for perfect alignment.
* **The Result:** Performance regressed to **7.61s** (much slower than 4.14s).
* **The Diagnosis:** **Occupancy Starvation**. By forcing `vector_length(32)`, we allocated only 1 warp per CUDA block. GPU hardware needs multiple warps (e.g., 4-8) per block to switch contexts and hide memory latency. Being "clever" with tiling backfired because we starved the scheduler.
* **Lesson:** Prefer `collapse(2)` and let the compiler auto-tune the vector length (it usually selects 128) for this specific algorithm.

---

## 📊 Performance Summary

| Implementation | Time (s) | Speedup | Status |
| --- | --- | --- | --- |
| **Baseline (Static)** | 12.26 s | 1.00x | Reference |
| **V1 (Basic Dynamic)** | 9.06 s | 1.35x | No Async |
| **V3 (Manual Tiling)** | 7.61 s | 1.61x | Occupancy Issue |
| **V4 (Final Opt)** | **4.14 s** | **2.96x** | **Bandwidth Limit** |

### The "Bandwidth Wall"

We cannot optimize further with standard methods.

* **Data Moved:** ~1.34 TB over 10k iterations.
* **Effective Speed:** 323 GB/s.
* **Hardware Peak:** 448 GB/s (RTX 2070).
* **Conclusion:** We are at 72% efficiency. The GPU is waiting on RAM, not computation.

---

## 🔮 Future Work

To break the 4.14s barrier, we would need advanced techniques beyond standard OpenACC directives:

1. **Temporal Blocking (Wavefronts):** Load a tile into Shared Memory (L1) and compute multiple time steps (t+1, t+2...) entirely inside the cache before writing back to Global Memory. This increases arithmetic intensity.
2. **Half-Precision (FP16):** Switching from `float` to `__half` would cut memory traffic by 50%, theoretically doubling speed to ~2.0s.
3. **Multi-GPU (MPI):** Decompose the grid across multiple GPUs using MPI + OpenACC to handle larger mesh sizes.

---

## 👥 Authors

* Joel Dettinger
* Harry Wolimba Hall
* Anna-Katharina Stsepankova
