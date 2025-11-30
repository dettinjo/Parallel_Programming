#!/bin/bash

#SBATCH --job-name=v1_profiling
#SBATCH --output=../results/v1_profile_%j.log
#SBATCH --error=../results/v1_profile_%j.err
#SBATCH --partition=nodo.q
#SBATCH --nodes=5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=1733432@uab.cat
#SBATCH --ntasks-per-node=12
#SBATCH --time=00:20:00

# Load modules
module load gcc/12.2.1
module load openmpi/4.1.1

# Setup paths
PROJECT_ROOT=$( cd -- "${SLURM_SUBMIT_DIR}/../" &> /dev/null && pwd )
SRC_DIR="${PROJECT_ROOT}/src"
RESULTS_DIR="${PROJECT_ROOT}/results"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Compile profiling version
mpicc -O3 -lm "${SRC_DIR}/laplace_mpi_v1_profiling.c" -o "${PROJECT_ROOT}/laplace_v1_profile"

# Test configurations for profiling
SIZES="1024 2048 4096 8192 16384 32768"
PROCESSES="4 8 12 24"

# Output file for profiling data
PROFILE_FILE="${RESULTS_DIR}/v1_performance_profile_${SLURM_JOB_ID}.csv"

echo "Starting V1 performance profiling..."

# Create CSV header
echo "matrix_size,processes,total_time,comm_time,comp_time,error_time,other_time,comm_percent,comp_percent,error_percent,efficiency,iterations" > "${PROFILE_FILE}"

# Run profiling experiments
for size in ${SIZES}; do
    for procs in ${PROCESSES}; do
        echo "Running: ${size}x${size} matrix with ${procs} processes"
        
        # Run and extract performance data
        output=$(mpirun -np ${procs} "${PROJECT_ROOT}/laplace_v1_profile" ${size} | grep "PERFORMANCE_ANALYSIS")
        
        if [[ -n "$output" ]]; then
            # Extract data from output (remove PERFORMANCE_ANALYSIS prefix)
            data=$(echo "$output" | sed 's/PERFORMANCE_ANALYSIS,//')
            echo "$data" >> "${PROFILE_FILE}"
        else
            echo "ERROR: No performance data for ${size}x${size}, ${procs} processes"
        fi
        
        # Small delay between runs
        sleep 2
    done
done

echo "V1 profiling complete. Results saved to: ${PROFILE_FILE}"

# Clean up
rm -f "${PROJECT_ROOT}/laplace_v1_profile"