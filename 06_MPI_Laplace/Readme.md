# 06_MPI_Laplace: Parallel Laplace Solver with MPI

This project implements a parallel solution for the 2D Laplace equation using the Message Passing Interface (MPI). It is designed to be compiled and executed on a SLURM-managed HPC cluster.

The repository includes:
* The sequential (baseline) C code.
* The parallel MPI C code.
* A set of robust SLURM automation scripts for compiling, testing, and running multi-node experiments.
* A Python script to automatically generate speedup plots from the experiment data.

## 📁 File Structure

```
06_MPI_Laplace/
├── src/
│   ├── laplace_seq.c
│   └── laplace_mpi.c
├── scripts/
│   ├── run_laplace_clus.job
│   ├── test_laplace_clus.job
│   ├── run_laplace_aolin.job
│   ├── test_laplace_aolin.job
│   ├── plot_results.py
│   └── run_plotter.job
└── results/
    ├── (Logs, CSVs, and plots are generated here)

```

## 🖥️ Cluster Environments

This project is pre-configured with SLURM scripts for two different clusters, "Clus" and "Aolin," each with unique partitions.

### Clus Cluster
* **Main Experiment (`run_laplace_clus.job`):**
    * **Partition:** `nodo.q`
    * **Resources:** 10 nodes, 12 cores/node (120 cores total)
* **Test Script (`test_laplace_clus.job`):**
    * **Partition:** `test.q`
    * **Resources:** 1 node, 8 cores (safe limit for this partition)

### Aolin Cluster
* **Main Experiment (`run_laplace_aolin.job`):**
    * **Partition:** `cuda-int.q` (Heterogeneous)
    * **Resources:** 2 nodes (1x 24-core, 1x 8-core). Script safely defaults to **8 cores/node** (16 cores total).
* **Test Script (`test_laplace_aolin.job`):**
    * **Partition:** `xeon.q`
    * **Resources:** 1 node, 8 cores.

## 🚀 How to Run an Experiment

Follow these steps to run a test or a full experiment. **All `sbatch` commands must be run from the `scripts/` directory.**

### 1. (Optional) Run a Test Job
It is highly recommended to run a test job first to verify compilation and basic functionality.

1.  Navigate to the scripts directory:
    ```bash
    cd "/home/master/ppm/ppm-43/Parallel Programming/06_MPI_Laplace/scripts"
    ```
2.  Submit the test script for your cluster:
    ```bash
    # On the Clus cluster:
    sbatch test_laplace_clus.job
    
    # On the Aolin cluster:
    sbatch test_laplace_aolin.job
    ```
3.  Check the output in the `results/` folder. You should see `laplace_test_...log` and `...csv` files.

### 2. Run a Full Experiment
Once testing is complete, run the main experiment script for the partition you want to use.

1.  Navigate to the scripts directory:
    ```bash
    cd ".../06_MPI_Laplace/scripts"
    ```
2.  Submit the desired experiment script:
    ```bash
    # On Clus (nodo.q, 12-core nodes):
    sbatch run_laplace_clus.job
    
    # On Aolin (cuda-int.q, 8-core minimum):
    sbatch run_laplace_aolin.job
    ```
3.  Monitor your job's status:
    ```bash
    squeue -u $USER
    ```

When the job is complete, you will find a unique `.log` file and a `.csv` file in the `results/` directory, tagged with the SLURM job ID (e.g., `laplace_timings_clus_70040.csv`).

## 📊 Generating Plots

After your experiment finishes, you can automatically generate a speedup plot.

### 1. (One-Time Only) Install Python Packages
If you have never run the plotter before, you must install the required libraries:
```bash
module load miniconda/3
pip install pandas matplotlib
```

### 2\. Run the Plotter Script

The `run_plotter.job` script will automatically find the **most recent** `laplace_timings_*.csv` file in the `results/` folder and create a matching plot.

1.  Navigate to the scripts directory:
    ```bash
    cd ".../06_MPI_Laplace/scripts"
    ```
2.  Submit the plotter job:
    ```bash
    sbatch run_plotter.job
    ```

This will create a new plot in the `results/` folder, such as `laplace_speedup_new-nodo_70040.png`.

<!-- end list -->
