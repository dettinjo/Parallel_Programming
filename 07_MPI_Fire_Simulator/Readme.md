# MPI Fire Simulator

This project contains the **MPI parallelization** of a 2D heat propagation and fire extinguishing simulation. It transforms a sequential C program into a distributed memory application capable of scaling across multiple nodes in a High-Performance Computing (HPC) cluster.

## Project Structure
```
.
├── src/
│   ├── fire_seq.c              # Sequential baseline code
│   ├── fire_mpi.c              # Parallel MPI implementation (Current: 2D Decomposition)
│   └── test_files/             # Input datasets (test1, test2, test3, test4)
├── scripts/
│   ├── test_fire_clus.job      # Main SLURM benchmarking script
│   ├── run_fire_plotter.job    # SLURM script to run the Python plotter
│   └── plot_fire_results.py    # Python script for speedup visualization
├── results/                    # Output logs, CSVs, and PNG plots
└── Readme.md
```
## Current State of the Project

### 1. Implementation Status (`src/fire_mpi.c`)
The codebase currently implements an advanced **2D Domain Decomposition** strategy.
* **Features:**
    * Uses `MPI_Dims_create` to split the global grid into a checkerboard pattern.
    * Implements **Manual Buffer Packing** to handle non-contiguous column data efficiently.
    * Uses **Asynchronous Communication** (`MPI_Isend`/`MPI_Irecv`) to overlap communication overhead with the computation of the inner grid.
    * Uses **Pointer Swapping** to eliminate $O(N)$ memory copying.
* **Performance:** Scales up to ~27x on 100 cores for the large dataset (`test2`), which is highly competitive but slightly slower than the Optimized 1D version due to the latency overhead of communicating with 4 neighbors on this specific cluster.

### 2. Known Constraints & Issues
* **Divisibility Constraint:** The simulation strictly requires that the grid dimensions be perfectly divisible by the process topology (e.g., rows % rows_in_grid == 0). This check exists in the legacy/read-only part of the code and causes an `MPI_Abort` if violated.
* **Scaling Limits:** For small datasets (`test1`, `test3`, `test4`), performance degrades beyond ~32 cores because the problem size per core becomes too small relative to the network latency.

## Automated Benchmarking Workflow

A robust set of scripts automates compilation, execution, timing, and plotting on the cluster.

### 1. Benchmarking Script: `scripts/test_fire_clus.job`

This is the primary script for running experiments.

**Usage:**
```bash
sbatch scripts/test_fire_clus.job
````

**Key Features:**

  * **Dynamic Pathing:** Automatically detects the project root, ensuring portability.
  * **Auto-Compilation:** Compiles both `fire_seq.c` and `fire_mpi.c` into temporary binaries at runtime to ensure the latest code is always tested.
  * **Validation Logic:** The script dynamically selects valid process counts for each test case to work around the "Divisibility Constraint".
      * *Example:* It runs 100 processes instead of 120 for `test2` because 8000 is not divisible by 120.
  * **Data Collection:** Parses the output of Rank 0 and saves structured performance data (Time, Speedup) to `results/fire_timings_clus_<JOB_ID>.csv`.

### 2\. Visualization: `scripts/run_fire_plotter.job`

A lightweight job that runs the Python plotting script on a compatible node.

**Usage:**

```bash
sbatch scripts/run_fire_plotter.job
```

**What it does:**

  * Loads necessary Python modules (`pandas`, `matplotlib`).
  * Reads the latest `.csv` file from the `results/` directory.
  * Generates speedup plots (e.g., `fire_speedup_clus_<JOB_ID>.png`) comparing the MPI performance against the sequential baseline.
  * Can be configured to generate a **Comparison Plot** merging multiple result files (Basic 1D vs Optimized 1D vs 2D) for the final report.

## How to Reproduce Results

1.  **Submit the Benchmark:**

    ```bash
    sbatch scripts/test_fire_clus.job
    ```

    Wait for the job to complete (check with `squeue`).

2.  **Generate Plots:**

    ```bash
    sbatch scripts/run_fire_plotter.job
    ```

3.  **View Results:**
    Check the `results/` folder for:

      * `.log` files: Raw output from the cluster.
      * `.csv` files: Parsed timing data.
      * `.png` files: Visualized speedup graphs.

