#!/bin/bash
#SBATCH --job-name=openmp_p5_FIXED
#SBATCH --output=openmp_p5_FIXED.out
#SBATCH --error=openmp_p5_FIXED.err

# --- Optimal Resource Request ---
#SBATCH --partition=cuda-ext.q       # Specify the partition
#SBATCH --nodelist=aolin-gpu-1       # Target the idle node
#SBATCH --nodes=1                    # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=2            # *** CRITICAL: Request only 2 CPUs ***
#SBATCH --time=00:10:00              # 10 mins

# --- Configuration ---
SRC_FILE="PrgPAR.c"
EXE_FILE="PrgPAR"
ARGS="20000 250000" # N=20000, REP=250000

# --- Compilation ---
echo "Compiling ${SRC_FILE}..."
gcc -Ofast -fno-inline -fopenmp ${SRC_FILE} -o ${EXE_FILE} -lm

if [ $? -ne 0 ]; then
  echo "Compilation failed!"
  exit 1
fi

echo "Compilation successful."

# --- Execution ---

# *** CRITICAL FIX ***
# Force the 'omp parallel' region to create a team of ONLY 2 threads.
# This eliminates barrier contention and fixes the race condition.
export OMP_NUM_THREADS=2

# We don't need nested parallelism for this code.
export OMP_NESTED=false

echo "========================================="
echo "Running on node $(hostname) with OMP_NUM_THREADS=2"
echo "========================================="
# We will use 'time' to get a simple wall-clock reading
time ./${EXE_FILE} $ARGS

echo "========================================="
echo "Analysis complete."