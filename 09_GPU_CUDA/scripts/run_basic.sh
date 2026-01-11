#!/bin/bash

#SBATCH --job-name=simple_bench
#SBATCH --output=simple_bench_%j.log
#SBATCH --error=simple_bench_%j.err
#SBATCH --partition=cuda-ext.q
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:30:00

echo "=== SIMPLE BENCHMARK (using working compilation from before) ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

# Use the EXACT same setup that worked before
if [[ "$SLURM_SUBMIT_DIR" == *"scripts" ]]; then
    PROJECT_ROOT="$(dirname "${SLURM_SUBMIT_DIR}")"
else
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
fi

SRC_DIR="${PROJECT_ROOT}/src"
JOB_ID=${SLURM_JOB_ID}

echo "Source dir: ${SRC_DIR}"
echo "Files available:"
ls -la ${SRC_DIR}/

# Load modules EXACTLY like before
module purge
module add nvhpc/21.2

echo "Loaded modules:"
module list

# Use the EXACT compilation flags that worked before
FLAGS="-fast -acc=gpu -gpu=cc60,cc70,cc75,cc80 -Minfo=accel"

echo "=== COMPILATION (using exact working commands) ==="

echo "1. Compiling OpenACC Baseline..."
nvc ${FLAGS} "${SRC_DIR}/laplace_baseline.c" -o "${SRC_DIR}/baseline_${JOB_ID}"
if [ $? -eq 0 ]; then echo "   ✓ Success"; else echo "   ✗ Failed"; fi

echo "2. Compiling CUDA v2..."
nvcc -O3 "${SRC_DIR}/laplace_cuda_v2.cu" -o "${SRC_DIR}/cuda_${JOB_ID}"
if [ $? -eq 0 ]; then echo "   ✓ Success"; else echo "   ✗ Failed"; fi

echo "3. Compiling OpenACC optimized..."
nvc ${FLAGS} "${SRC_DIR}/laplace_openacc.c" -o "${SRC_DIR}/openacc_${JOB_ID}"
if [ $? -eq 0 ]; then echo "   ✓ Success"; else echo "   ✗ Failed"; fi

# Skip MPI and OpenMP for now since they're causing issues
echo "Skipping MPI and OpenMP for now..."

echo "=== SIMPLE TEST RUN ==="
SIZE=2048
ITERS=1000

echo "Testing ${SIZE}x${SIZE} matrix with ${ITERS} iterations:"

# Test baseline
if [ -f "${SRC_DIR}/baseline_${JOB_ID}" ]; then
    echo -n "OpenACC Baseline: "
    start=$(date +%s.%N)
    result=$("${SRC_DIR}/baseline_${JOB_ID}" ${SIZE} ${SIZE} ${ITERS} 2>&1)
    end=$(date +%s.%N)
    wall_time=$(echo "$end - $start" | bc)
    echo "${wall_time}s (program reported: ${result}s)"
fi

# Test CUDA
if [ -f "${SRC_DIR}/cuda_${JOB_ID}" ]; then
    echo -n "CUDA v2: "
    start=$(date +%s.%N)
    result=$("${SRC_DIR}/cuda_${JOB_ID}" ${SIZE} ${SIZE} ${ITERS} 2>&1)
    end=$(date +%s.%N)
    wall_time=$(echo "$end - $start" | bc)
    echo "${wall_time}s (program reported: ${result}s)"
fi

# Test OpenACC optimized
if [ -f "${SRC_DIR}/openacc_${JOB_ID}" ]; then
    echo -n "OpenACC optimized: "
    start=$(date +%s.%N)
    result=$("${SRC_DIR}/openacc_${JOB_ID}" ${SIZE} ${SIZE} ${ITERS} 2>&1)
    end=$(date +%s.%N) 
    wall_time=$(echo "$end - $start" | bc)
    echo "${wall_time}s (program reported: ${result}s)"
fi

echo "=== CLEANUP ==="
rm -f "${SRC_DIR}/baseline_${JOB_ID}" "${SRC_DIR}/cuda_${JOB_ID}" "${SRC_DIR}/openacc_${JOB_ID}"

echo "Done."