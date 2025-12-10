# Parallel Programming - Course Repository

This repository contains all the assignments, laboratory work, and reports for the **Parallel Programming** course, part of the **Master on Modelling for Science and Engineering** at the **Universitat Autònoma de Barcelona (UAB)**.

**Instructors:**
*   Juan Carlos Moure (`juancarlos.moure@uab.es`)
*   Sandra Mendez (`sandra.mendez@uab.es`)
*   Christian Guzmán (`christian.guzman@uab.es`)

## Team Members (Group 22)

This project was developed by:
*   **Joel Dettinger**
*   **Harry Wolimba Hall**
*   **Anna-Katharina Stsepankova**

---

## Course Overview

This course provides a comprehensive introduction to the fundamental concepts and technologies of high-performance and parallel computing. The main objective is to understand how to solve larger, more complex problems by leveraging parallel architectures.

Key topics covered include:
*   **C Programming & Performance Analysis:** Writing, benchmarking, and optimizing scientific code.
*   **Parallelism Concepts:** Understanding data locality, communication, synchronization, and sources of inefficiency.
*   **Parallel Technologies:**
    *   **Multicore Processors:** OpenMP
    *   **Distributed Systems:** MPI
    *   **Accelerators (GPUs):** CUDA / OpenACC

## Deliverables

This repository is structured by course deliverables. According to the evaluation criteria, the deliverables are:
*   **[20%] C Programming & Performance:** Implementation and analysis of a Jacobi solver for the Laplace equation.
*   **[15%] OpenMP:** Parallelizing applications for shared-memory multicore processors.
*   **[20%] MPI:** Developing applications for distributed-memory systems and clusters.
*   **[15%] GPU Programming:** Offloading computations to accelerators.

### Project 1: C Programming - Jacobi Solver for the 2D Laplace Equation

This project serves as a foundation for the course, focusing on single-core performance, memory management, and optimization techniques before introducing parallelism. The goal is to solve the 2D Laplace equation using an iterative Jacobi method.

The project is divided into three distinct implementations to analyze performance trade-offs:
1.  **V1 - Static Version:** A simple implementation where the computational grid size is fixed at compile-time (`#define`). This version requires recompilation for each problem size.
2.  **V2 - Dynamic Version:** A more flexible version that uses dynamic memory allocation (`malloc`) to allow the user to specify the grid size at runtime via command-line arguments.
3.  **V3 - Optimized Version:** An enhanced version of the dynamic code that incorporates both compiler-level optimizations (`-O2` flag) and source-code level improvements (**loop fusion**) to improve data locality and reduce execution time.

## HPC Environment Setup

All benchmarks and tests are designed to be run on a high-performance computing (HPC) cluster environment.
*   **Clusters:** `aolin` and `Vilma`
*   **Operating System:** Linux
*   **Queue Manager:** Slurm Workload Manager
*   **Connection:** Remote connection via SSH (e.g., using MobaXterm, Windows Terminal, or a native terminal).

## How to Compile and Run Project 1

This section provides instructions to replicate the results presented in the report for the Jacobi solver.

### Prerequisites
*   A C compiler (e.g., GCC)
*   Access to a Slurm-managed Linux cluster
*   Python 3 with `matplotlib` and `numpy` for generating the performance chart (`pip install matplotlib numpy`)
*   A LaTeX distribution (e.g., TeX Live, MiKTeX) for compiling the final report.

### Compilation

Navigate to the project directory and compile the three C versions. Note that the static version requires a separate compilation for each grid size.

```bash
# Compile the Dynamic version (V2)
gcc dyn.c -o dyn -lm

# Compile the Optimized version (V3) with -O2 flag
gcc -O2 opt.c -o opt -lm

# Compile the Static versions (V1) for each specific size
gcc -DN=100 -DM=100 stat.c -o stat_100 -lm
gcc -DN=1000 -DM=1000 stat.c -o stat_1000 -lm
gcc -DN=4096 -DM=4096 stat.c -o stat_4096 -lm
```

### Execution

The entire benchmarking process is automated using a Slurm script. This script runs all nine test cases, captures their `real` (wall-clock) time, and generates a comprehensive report file (`laplace_report.txt`) containing the raw output and a final summary table.

1.  **Ensure all executables** (`dyn`, `opt`, `stat_100`, `stat_1000`, `stat_4096`) are in the same directory as the Slurm script.

2.  **Submit the job to the Slurm queue manager:**
    ```bash
    sbatch run_and_summarize.slurm
    ```

3.  **Monitor the job status:**
    ```bash
    squeue -u <your_username>
    ```

4.  **Check the results:** Once the job is complete, the results will be in the file `laplace_report.txt`. A Slurm log file (e.g., `slurm_log_<job_id>.out`) will also be created.

### Generating the Report Figure

An accessible, color-blind friendly performance chart can be generated using the provided Python script.
```bash
python create_accessible_figure.py
```
This will create `performance_chart_accessible.png`, which is used in the final LaTeX report.