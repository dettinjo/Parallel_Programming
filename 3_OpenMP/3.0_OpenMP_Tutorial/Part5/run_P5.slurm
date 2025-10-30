#!/bin/bash
#SBATCH --job-name=openmp_p5_TASK
#SBATCH --output=openmp_p5_TASK.out
#SBATCH --error=openmp_p5_TASK.err
#SBATCH --partition=cuda-ext.q
#SBATCH --nodelist=aolin-gpu-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:10:00

# --- Configuration ---
SRC_FILE="PrgPAR_P5.c"
EXE_FILE="PrgPAR_P5"
ARGS="20000 250000"

# --- Compilation ---
echo "Compiling ${SRC_FILE}..."
# We can remove -fno-inline now.
gcc -Ofast -fopenmp ${SRC_FILE} -o ${EXE_FILE} -lm
echo "Compilation successful."

# --- Execution ---
export OMP_NUM_THREADS=12
export OMP_NESTED=false
# We use 'active' (default) because threads are
# now busy with tasks, not spin-waiting at barriers.
export OMP_WAIT_POLICY=active

echo "========================================="
echo "Running on node $(hostname) with OMP_NUM_THREADS=12"
echo "OMP_WAIT_POLICY=${OMP_WAIT_POLICY}"
echo "========================================="
time ./${EXE_FILE} $ARGS
echo "========================================="
echo "Analysis complete."