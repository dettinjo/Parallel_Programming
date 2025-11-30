#!/bin/bash
# run_weak_scaling.sh - Quick weak scaling analysis
# Maintains ~1024x1024 elements per process

#SBATCH --job-name=weak_scaling
#SBATCH --output=../results/weak_scaling_%j.log
#SBATCH --partition=nodo.q
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=12
#SBATCH --time=00:15:00

PROJECT_ROOT=$( cd -- "${SLURM_SUBMIT_DIR}/../" &> /dev/null && pwd )
BIN_MPI="${PROJECT_ROOT}/laplace_mpi_bin"
CSV_FILE="../results/weak_scaling_${SLURM_JOB_ID}.csv"

# Load modules
module load gcc/12.2.1
module load openmpi/4.1.1

# Compile if needed
mpicc -O3 -lm /mnt/project/laplace_mpi_v1.c -o "${BIN_MPI}"

echo "processes,matrix_size,elements_per_proc,time_seconds,efficiency" > "${CSV_FILE}"

echo "=== WEAK SCALING ANALYSIS ==="
echo "Maintaining ~1048576 elements per process"

# Weak scaling configurations (approximate)
# 1 proc: 1024x1024 = 1,048,576 elements
echo "Testing 1 process, 1024x1024..."
TIME_1=$(mpirun -np 1 "${BIN_MPI}" 1024 | head -n 1)
ELEMENTS_1=1048576
echo "1,1024,$ELEMENTS_1,$TIME_1,1.00" >> "${CSV_FILE}"
echo "1 process: ${TIME_1}s"

# 4 procs: 2048x2048 = 4,194,304 elements = ~1M per proc  
echo "Testing 4 processes, 2048x2048..."
TIME_4=$(mpirun -np 4 "${BIN_MPI}" 2048 | head -n 1)
ELEMENTS_4=1048576
EFF_4=$(awk "BEGIN {printf \"%.3f\", $TIME_1 / $TIME_4}")
echo "4,2048,$ELEMENTS_4,$TIME_4,$EFF_4" >> "${CSV_FILE}"
echo "4 processes: ${TIME_4}s, efficiency: ${EFF_4}"

# 16 procs: 4096x4096 = 16,777,216 elements = ~1M per proc
echo "Testing 16 processes, 4096x4096..."
TIME_16=$(mpirun -np 16 "${BIN_MPI}" 4096 | head -n 1)
ELEMENTS_16=1048576  
EFF_16=$(awk "BEGIN {printf \"%.3f\", $TIME_1 / $TIME_16}")
echo "16,4096,$ELEMENTS_16,$TIME_16,$EFF_16" >> "${CSV_FILE}"
echo "16 processes: ${TIME_16}s, efficiency: ${EFF_16}"

# 64 procs: 8192x8192 = 67,108,864 elements = ~1M per proc
echo "Testing 64 processes, 8192x8192..."
TIME_64=$(mpirun -np 64 "${BIN_MPI}" 8192 | head -n 1)
ELEMENTS_64=1048576
EFF_64=$(awk "BEGIN {printf \"%.3f\", $TIME_1 / $TIME_64}")
echo "64,8192,$ELEMENTS_64,$TIME_64,$EFF_64" >> "${CSV_FILE}"
echo "64 processes: ${TIME_64}s, efficiency: ${EFF_64}"

echo "=== WEAK SCALING COMPLETE ==="
echo "Results saved to: ${CSV_FILE}"