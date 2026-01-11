#!/bin/bash

#SBATCH --job-name=lap_benchmark_1000
#SBATCH --output=lap_benchmark_1000_%j.log
#SBATCH --error=lap_benchmark_1000_%j.err
#SBATCH --partition=cuda-ext.q
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4

# --- Path Setup ---
if [[ "$SLURM_SUBMIT_DIR" == *"scripts" ]]; then
    PROJECT_ROOT="$(dirname "${SLURM_SUBMIT_DIR}")"
else
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
fi

SRC_DIR="${PROJECT_ROOT}/src"
RESULTS_DIR="${PROJECT_ROOT}/results"
LOG_DIR="${RESULTS_DIR}/log"
ERR_DIR="${RESULTS_DIR}/err"
CSV_DIR="${RESULTS_DIR}/csv"

mkdir -p "${LOG_DIR}" "${ERR_DIR}" "${CSV_DIR}"

JOB_ID=${SLURM_JOB_ID}
CSV_FILE="${CSV_DIR}/benchmark_1000iter_${SLURM_JOB_PARTITION}_${JOB_ID}.csv"
DETAILED_CSV="${CSV_DIR}/benchmark_1000iter_detailed_${SLURM_JOB_PARTITION}_${JOB_ID}.csv"

# Binary names
BIN_BASELINE="${SRC_DIR}/laplace_baseline_${JOB_ID}"
BIN_CUDA="${SRC_DIR}/laplace_cuda_${JOB_ID}"
BIN_MPI="${SRC_DIR}/laplace_mpi_${JOB_ID}"
BIN_OMP="${SRC_DIR}/laplace_omp_${JOB_ID}"
BIN_OPENACC="${SRC_DIR}/laplace_openacc_${JOB_ID}"

cleanup() {
    rm -f "${BIN_BASELINE}" "${BIN_CUDA}" "${BIN_MPI}" "${BIN_OMP}" "${BIN_OPENACC}"
    sleep 2
    mv "${SLURM_SUBMIT_DIR}/lap_benchmark_1000_${JOB_ID}.log" "${LOG_DIR}/" 2>/dev/null
    mv "${SLURM_SUBMIT_DIR}/lap_benchmark_1000_${JOB_ID}.err" "${ERR_DIR}/" 2>/dev/null
}
trap cleanup EXIT

echo "=== CONSISTENT 1000-ITERATION LAPLACE BENCHMARK (${JOB_ID}) ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo

# --- Load Modules ---
module purge
module add nvhpc/21.2
module add openmpi/4.1.1

# --- CONSISTENT COMPILATION FLAGS ---
echo "=== Compilation Phase ==="
FAST_FLAGS="-fast"
GPU_FLAGS="-fast -acc=gpu -gpu=cc60,cc70,cc75,cc80"
OMP_FLAGS="-fast -mp"
CUDA_FLAGS="-O3 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70"

COMPILE_SUCCESS=0

echo "1. Compiling OpenACC Baseline..."
nvc ${GPU_FLAGS} "${SRC_DIR}/laplace_baseline.c" -o "${BIN_BASELINE}" 2>&1
if [ $? -eq 0 ]; then echo "   ✓ Success"; COMPILE_SUCCESS=$((COMPILE_SUCCESS + 1)); else echo "   ✗ Failed"; fi

echo "2. Compiling CUDA v2..."
nvcc ${CUDA_FLAGS} "${SRC_DIR}/laplace_cuda_v2.cu" -o "${BIN_CUDA}" 2>&1
if [ $? -eq 0 ]; then echo "   ✓ Success"; COMPILE_SUCCESS=$((COMPILE_SUCCESS + 1)); else echo "   ✗ Failed"; fi

echo "3. Compiling MPI v1 (FIXED)..."
mpicc ${FAST_FLAGS} "${SRC_DIR}/laplace_mpi_v1.c" -o "${BIN_MPI}" -lm 2>&1
if [ $? -eq 0 ]; then echo "   ✓ Success"; COMPILE_SUCCESS=$((COMPILE_SUCCESS + 1)); else echo "   ✗ Failed"; fi

echo "4. Compiling OpenMP v3..."
nvc ${OMP_FLAGS} "${SRC_DIR}/laplace_omp_v3.c" -o "${BIN_OMP}" -lm 2>&1
if [ $? -eq 0 ]; then echo "   ✓ Success"; COMPILE_SUCCESS=$((COMPILE_SUCCESS + 1)); else echo "   ✗ Failed"; fi

echo "5. Compiling OpenACC optimized..."
nvc ${GPU_FLAGS} "${SRC_DIR}/laplace_openacc.c" -o "${BIN_OPENACC}" 2>&1
if [ $? -eq 0 ]; then echo "   ✓ Success"; COMPILE_SUCCESS=$((COMPILE_SUCCESS + 1)); else echo "   ✗ Failed"; fi

echo "Compilation: ${COMPILE_SUCCESS}/5 successful"
echo

# --- BENCHMARK CONFIGURATION ---
SIZES=(2048 4096 8192)          # Matrix sizes to test
ITERATIONS=1000                  # CONSISTENT iterations for ALL implementations
RUNS=3                          # Multiple runs for statistical reliability

echo "=== Benchmark Configuration ==="
echo "Matrix sizes: ${SIZES[@]}"
echo "Iterations per test: ${ITERATIONS}"
echo "Runs per test: ${RUNS}"
echo

# Initialize CSV files
echo "implementation,matrix_size,run,time_seconds,speedup_vs_baseline" > "${DETAILED_CSV}"
echo "implementation,matrix_size,avg_time,min_time,max_time,avg_speedup" > "${CSV_FILE}"

# --- BENCHMARK FUNCTION ---
run_benchmark() {
    local impl_name=$1
    local binary_path=$2
    local size=$3
    local is_mpi=$4
    
    if [ ! -f "${binary_path}" ]; then
        echo "   Binary not found: ${binary_path}"
        return 1
    fi
    
    echo "   ${impl_name} (${size}x${size}, ${ITERATIONS} iterations):"
    
    local times=()
    local successful_runs=0
    
    for run in $(seq 1 ${RUNS}); do
        echo -n "     Run ${run}/${RUNS}... "
        
        local start_time=$(date +%s.%N)
        
        # Execute with CONSISTENT parameters: matrix_width matrix_height iterations
        if [ "${is_mpi}" == "true" ]; then
            # MPI: Now accepts iterations as 3rd parameter (FIXED!)
            local output=$(timeout 120s mpirun -np 4 "${binary_path}" ${size} ${size} ${ITERATIONS} 2>&1)
            local exit_code=$?
        else
            # All other implementations: width height iterations
            local output=$(timeout 120s "${binary_path}" ${size} ${size} ${ITERATIONS} 2>&1)
            local exit_code=$?
        fi
        
        local end_time=$(date +%s.%N)
        local elapsed=$(echo "$end_time - $start_time" | bc -l)
        
        if [ ${exit_code} -eq 0 ]; then
            times+=(${elapsed})
            successful_runs=$((successful_runs + 1))
            printf "%.4fs ✓\n" "${elapsed}"
        else
            printf "FAILED (exit: %d)\n" "${exit_code}"
            echo "       Output: ${output}"
        fi
    done
    
    if [ ${successful_runs} -eq 0 ]; then
        echo "     All runs failed!"
        return 1
    fi
    
    # Calculate statistics
    local sum=0
    local min_time=${times[0]}
    local max_time=${times[0]}
    
    for time in "${times[@]}"; do
        sum=$(echo "$sum + $time" | bc -l)
        min_time=$(echo "if ($time < $min_time) $time else $min_time" | bc -l)
        max_time=$(echo "if ($time > $max_time) $time else $max_time" | bc -l)
    done
    
    local avg_time=$(echo "scale=6; $sum / ${#times[@]}" | bc -l)
    
    echo "     Results: avg=${avg_time}s, min=${min_time}s, max=${max_time}s"
    
    # Store results for speedup calculation
    eval "${impl_name}_${size}_avg=${avg_time}"
    eval "${impl_name}_${size}_min=${min_time}"
    eval "${impl_name}_${size}_max=${max_time}"
    eval "${impl_name}_${size}_success=true"
    
    return 0
}

# --- EXECUTE BENCHMARKS ---
echo "=== Running Benchmarks ==="

# Define implementations in test order
implementations=(
    "baseline:${BIN_BASELINE}:false"
    "cuda_v2:${BIN_CUDA}:false"  
    "mpi_v1:${BIN_MPI}:true"
    "omp_v3:${BIN_OMP}:false"
    "openacc:${BIN_OPENACC}:false"
)

for size in "${SIZES[@]}"; do
    echo "Matrix Size: ${size}x${size}"
    echo "=================================="
    
    for impl_info in "${implementations[@]}"; do
        IFS=':' read -r impl_name binary_path is_mpi <<< "${impl_info}"
        run_benchmark "${impl_name}" "${binary_path}" "${size}" "${is_mpi}"
    done
    echo
done

# --- CALCULATE SPEEDUPS ---
echo "=== Speedup Analysis ==="
printf "%-12s %-8s %-12s %-12s\n" "Implementation" "Size" "Time (s)" "Speedup"
printf "%-12s %-8s %-12s %-12s\n" "==============" "====" "========" "======="

for size in "${SIZES[@]}"; do
    # Get baseline time
    baseline_avg="baseline_${size}_avg"
    baseline_time=${!baseline_avg}
    baseline_success="baseline_${size}_success"
    
    if [ "${!baseline_success}" == "true" ]; then
        printf "%-12s %-8s %-12.4f %-12s\n" "baseline" "${size}" "${baseline_time}" "1.00x"
        echo "baseline,${size},${baseline_time},${!baseline_min},${!baseline_max},1.00" >> "${CSV_FILE}"
        
        # Calculate speedups for other implementations
        for impl_info in "${implementations[@]}"; do
            IFS=':' read -r impl_name _ _ <<< "${impl_info}"
            
            if [ "${impl_name}" != "baseline" ]; then
                avg_var="${impl_name}_${size}_avg"
                min_var="${impl_name}_${size}_min"
                max_var="${impl_name}_${size}_max"
                success_var="${impl_name}_${size}_success"
                
                if [ "${!success_var}" == "true" ]; then
                    local speedup=$(echo "scale=2; ${baseline_time} / ${!avg_var}" | bc -l)
                    printf "%-12s %-8s %-12.4f %-12.2fx\n" "${impl_name}" "${size}" "${!avg_var}" "${speedup}"
                    echo "${impl_name},${size},${!avg_var},${!min_var},${!max_var},${speedup}" >> "${CSV_FILE}"
                else
                    printf "%-12s %-8s %-12s %-12s\n" "${impl_name}" "${size}" "FAILED" "N/A"
                    echo "${impl_name},${size},FAILED,FAILED,FAILED,FAILED" >> "${CSV_FILE}"
                fi
            fi
        done
        echo
    else
        echo "WARNING: Baseline failed for size ${size}x${size} - cannot calculate speedups"
    fi
done

echo "=== Summary ==="
echo "Results saved to:"
echo "  Summary:  ${CSV_FILE}"
echo "  Detailed: ${DETAILED_CSV}"
echo
echo "All implementations tested with exactly ${ITERATIONS} iterations for fair comparison."
echo "MPI implementation now uses consistent parameter structure."