#!/bin/bash
#SBATCH --job-name=laplace-perf
#SBATCH --output=laplace-perf-out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cuda-ext.q

# Compile the programs
gcc -std=c99 -Wall -O3 -march=native -o laplace_seq laplace_seq.c -lm
gcc -std=c99 -Wall -O3 -march=native -fopenmp -o laplace_omp_v1 laplace_omp_v1.c -lm

echo "=== Laplace Solver Performance Comparison ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Test different matrix sizes and thread counts
sizes=(500 1000 1500)
thread_counts=(1 2 4 8)

for size in "${sizes[@]}"; do
    echo "Matrix size: ${size}x${size}"
    echo ""
    
    # Sequential baseline
    echo "Sequential:"
    seq_output=$(./laplace_seq $size $size 2>/dev/null)
    seq_time=$(echo "$seq_output" | grep "Execution time" | awk '{print $3}')
    echo "Time: ${seq_time}s"
    echo ""
    
    # OpenMP with different thread counts
    for threads in "${thread_counts[@]}"; do
        export OMP_NUM_THREADS=$threads
        echo "OpenMP (${threads} threads):"
        omp_output=$(./laplace_omp_v1 $size $size 2>/dev/null)
        omp_time=$(echo "$omp_output" | grep "Execution time" | awk '{print $3}')
        
        if [ -n "$seq_time" ] && [ -n "$omp_time" ]; then
            speedup=$(echo "$seq_time $omp_time" | awk '{printf "%.2f", $1/$2}')
            echo "Time: ${omp_time}s, Speedup: ${speedup}x"
        else
            echo "Time: ${omp_time}s, Speedup: N/A"
        fi
    done
    echo ""
done