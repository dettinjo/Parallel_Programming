#!/bin/bash
#SBATCH --job-name=openmp_P5_solution
#SBATCH --output=openmp_results_P5_%j.out
#SBATCH --error=openmp_results_P5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --partition=cuda-ext.q
#SBATCH --nodelist=aolin-gpu-1

echo "========================================="
echo "OpenMP P5 Manual Solution vs Sequential Baseline Test"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================="
echo ""

# Compile both versions
echo "Compiling programs..."

# Check if source files exist
if [ ! -f "PrgPAR_P5.c" ]; then
    echo "Error: PrgPAR_P5.c not found!"
    exit 1
fi
if [ ! -f "Prg_baseline_timed.c" ]; then
    echo "Error: Prg_baseline_timed.c not found!"
    exit 1
fi

# Compile P5 optimized version
# *** FIX: Moved -fopenmp and -lm to the end, after the source file ***
gcc -O3 -march=native -std=c99 PrgPAR_P5.c -o PrgPAR_P5 -lm -fopenmp
if [ $? -ne 0 ]; then echo "P5 Solution compilation failed!"; exit 1; fi

# Compile sequential baseline version
# *** FIX: Moved -fopenmp and -lm to the end, after the source file ***
gcc -O3 -march=native Prg_baseline_timed.c -o Prg_baseline -lm -fopenmp
if [ $? -ne 0 ]; then echo "Baseline compilation failed!"; exit 1; fi


echo "Compilation successful!"
echo ""

# Test parameters
N=20000
REP=250000
SEQ_BASELINE_TIME=""
PAR_BASELINE_TIME="" # This will be the 1-thread parallel run

echo "========================================="
echo "Test Configuration:"
echo "  N = $N"
echo "  REP = $REP"
echo "========================================="
echo ""

# --- Run Sequential Baseline ---
echo "========================================="
echo "Running Sequential Baseline (Prg_baseline)"
echo "========================================="
output_seq=$(./Prg_baseline $N $REP 2>&1)
echo "$output_seq"
# Extract time and save the full output line
SEQ_BASELINE_OUTPUT=$(echo "$output_seq" | grep "Outputs:")
SEQ_BASELINE_TIME=$(echo "$output_seq" | grep "Execution time:" | awk '{print $3}')
if [ -z "$SEQ_BASELINE_TIME" ]; then
    echo "Error: Could not extract sequential baseline time."
    exit 1
fi
echo "Sequential Baseline Time: ${SEQ_BASELINE_TIME}s"
echo "Sequential Baseline Output: ${SEQ_BASELINE_OUTPUT}"
echo ""


# Function to calculate speedup
calculate_speedup() {
    local time=$1
    local threads=$2
    
    if [ ! -z "$PAR_BASELINE_TIME" ]; then
        # Use awk for floating point calculations
        local speedup_par=$(awk "BEGIN {print $PAR_BASELINE_TIME / $time}")
        local efficiency=$(awk "BEGIN {print $speedup_par / $threads * 100}")
        printf "  Speedup vs 1-thread parallel: %.2fx\n" $speedup_par
        printf "  Parallel Efficiency:          %.2f%%\n" $efficiency
    fi
    
    if [ ! -z "$SEQ_BASELINE_TIME" ]; then
        local speedup_seq=$(awk "BEGIN {print $SEQ_BASELINE_TIME / $time}")
        printf "  Speedup vs sequential code:   %.2fx\n" $speedup_seq
    fi
}

# Test with different thread counts
echo "========================================="
echo "Performance Testing with P5 Solution (PrgPAR_P5)"
echo "========================================="
echo ""

for threads in 1 2 4 6 8 12; do
    echo "--- Testing with $threads threads ---"
    export OMP_NUM_THREADS=$threads
    
    # Run and capture output
    output_par=$(./PrgPAR_P5 $N $REP 2>&1)
    echo "$output_par"
    
    # Extract execution time
    exec_time_par=$(echo "$output_par" | grep "Execution time:" | awk '{print $3}')
    
    if [ -z "$exec_time_par" ]; then
        echo "Error: Could not extract parallel execution time."
        continue
    fi
    
    # Store baseline time from 1 thread run
    if [ $threads -eq 1 ]; then
        PAR_BASELINE_TIME=$exec_time_par
        echo "  (Parallel 1-thread baseline for this run)"
    fi
    
    # Calculate speedups
    calculate_speedup $exec_time_par $threads
    echo ""
done

echo "========================================="
echo "Detailed Performance Analysis (6 threads)"
echo "========================================="

export OMP_NUM_THREADS=6

if command -v perf &> /dev/null; then
    echo "Running perf stat on P5 solution..."
    perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-load-misses,branches,branch-misses ./PrgPAR_P5 $N $REP
else
    echo "perf not available, running standard execution..."
    ./PrgPAR_P5 $N $REP
fi

echo ""
echo "========================================="
echo "Performance Summary"
echo "========================================="
echo "Sequential Baseline Time (Prg_baseline): ${SEQ_BASELINE_TIME}s"
echo "Parallel 1-Thread Time (PrgPAR_P5): ${PAR_BASELINE_TIME}s"
echo ""
echo "Sequential Baseline Output: ${SEQ_BASELINE_OUTPUT}"
echo "(Note: P5 solution output values are expected to differ)"
echo ""
echo "Job completed at: $(date)"
echo "========================================="

