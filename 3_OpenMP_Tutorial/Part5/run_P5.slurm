#!/bin/bash
#SBATCH --job-name=openmp_opt
#SBATCH --output=openmp_results_%j.out
#SBATCH --error=openmp_results_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --partition=cuda-ext.q

# Load necessary modules (adjust based on your cluster)
# module load gcc/latest
# module load intel/latest

echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "========================================="

# Compile the program
echo "Compiling the program..."

# Check if source file exists
if [ ! -f "PrgPAR_P5.c" ]; then
    echo "Error: PrgPAR_P5.c not found!"
    echo "Looking for C files in current directory:"
    ls -la *.c 2>/dev/null || echo "No .c files found"
    exit 1
fi

gcc -Ofast -fno-inline -fopenmp -march=native PrgPAR_P5.c -o PrgPAR_P5 -lm

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# Verify executable was created
if [ ! -f "./PrgPAR_P5" ]; then
    echo "Error: Executable was not created!"
    ls -la
    exit 1
fi

echo "Compilation successful!"
echo ""

# Test with different thread counts
N=20000
REP=250000

echo "========================================="
echo "Performance Analysis with Different Thread Counts"
echo "N=$N, REP=$REP"
echo "========================================="
echo ""

# Sequential baseline (if available)
if [ -f "Prg" ]; then
    echo "--- Sequential Baseline ---"
    time ./Prg $N $REP
    echo ""
fi

# Test with 2 threads
echo "--- Running with 2 threads ---"
export OMP_NUM_THREADS=2
time ./PrgPAR_P5 $N $REP 2
echo ""

# Test with 4 threads
echo "--- Running with 4 threads ---"
export OMP_NUM_THREADS=4
time ./PrgPAR_P5 $N $REP 4
echo ""

# Test with 6 threads (optimal for 6-core processor)
echo "--- Running with 6 threads ---"
export OMP_NUM_THREADS=6
time ./PrgPAR_P5 $N $REP 6
echo ""

# Test with 12 threads (hyperthreading)
echo "--- Running with 12 threads ---"
export OMP_NUM_THREADS=12
time ./PrgPAR_P5 $N $REP 12
echo ""

# Performance profiling with perf (if available)
echo "========================================="
echo "Detailed Performance Analysis (6 threads)"
echo "========================================="
export OMP_NUM_THREADS=6

if command -v perf &> /dev/null; then
    echo "Running perf stat..."
    perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses ./PrgPAR_P5 $N $REP 6
    echo ""
    
    echo "Running perf record..."
    perf record -g ./PrgPAR_P5 $N $REP 6
    echo "Generating perf report..."
    perf report --stdio > perf_report_6threads.txt
    echo "Report saved to perf_report_6threads.txt"
else
    echo "perf not available, skipping profiling"
fi

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="

# Generate summary
echo ""
echo "Summary of results saved in this output file"
echo "Check the speedup relative to the baseline"