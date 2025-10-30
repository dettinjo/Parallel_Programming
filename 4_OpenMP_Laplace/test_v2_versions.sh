#!/bin/bash
#SBATCH --job-name=laplace-v2-test
#SBATCH --output=laplace-v2-out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cuda-ext.q
#SBATCH --exclusive

# Compile all versions
gcc -std=c99 -Wall -O3 -march=native -o laplace_seq laplace_seq.c -lm
gcc -std=c99 -Wall -O3 -march=native -fopenmp -o laplace_omp_v1 laplace_omp_v1.c -lm
gcc -std=c99 -Wall -O3 -march=native -fopenmp -o laplace_omp_v2 laplace_omp_v2.c -lm
gcc -std=c99 -Wall -O3 -march=native -fopenmp -o laplace_omp_v2_simple laplace_omp_v2_simple.c -lm
#gcc -std=c99 -Wall -O3 -march=native -fopenmp -o laplace_omp_v2_swap laplace_omp_v2_swap.c -lm

echo "=== Laplace Solver v2 Performance Comparison ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Test sizes
sizes=(500 1000 1500)
thread_counts=(1 2 6 12 24)

for size in "${sizes[@]}"; do
    echo "Matrix size: ${size}x${size}"
    echo ""
    
    # Sequential baseline
    echo "Sequential:"
    seq_output=$(./laplace_seq $size $size 2>/dev/null)
    seq_time=$(echo "$seq_output" | grep "Execution time" | awk '{print $3}')
    echo "Time: ${seq_time}s"
    echo ""
    
    for threads in "${thread_counts[@]}"; do
        export OMP_NUM_THREADS=$threads
        echo "_______________________________"
        echo "OpenMP (${threads} threads):"
        echo "________________________________"

        v1_output=$(./laplace_omp_v1 $size $size 2>/dev/null)
        v1_time=$(echo "$v1_output" | grep "Execution time" | awk '{print $3}')
        v1_speedup=$(echo "$seq_time $v1_time" | awk '{printf "%.2f", $1/$2}')
        echo "Time: ${v1_time}s, Speedup: ${v1_speedup}x"
        echo ""

        v2_output=$(./laplace_omp_v2 $size $size 2>/dev/null)
        v2_time=$(echo "$v2_output" | grep "Execution time" | awk '{print $3}')
        v2_speedup=$(echo "$seq_time $v2_time" | awk '{printf "%.2f", $1/$2}')
        echo "Time: ${v2_time}s, Speedup: ${v2_speedup}x"
        echo ""

        v2s_output=$(./laplace_omp_v2_simple $size $size 2>/dev/null)
        v2s_time=$(echo "$v2s_output" | grep "Execution time" | awk '{print $3}')
        v2s_speedup=$(echo "$seq_time $v2s_time" | awk '{printf "%.2f", $1/$2}')
        echo "Time: ${v2s_time}s, Speedup: ${v2s_speedup}x"
        echo ""

        # v2p_output=$(./laplace_omp_v2_swap $size $size 2>/dev/null)
        # v2p_time=$(echo "$v2p_output" | grep "Execution time" | awk '{print $3}')
        # v2p_speedup=$(echo "$seq_time $v2p_time" | awk '{printf "%.2f", $1/$2}')
        # echo "Time: ${v2p_time}s, Speedup: ${v2p_speedup}x"
        # echo ""
    
        echo "--- Summary for ${size}x${size} ---"
        echo "Sequential: ${seq_time}s (baseline)"
        echo "v1:         ${v1_time}s (${v1_speedup}x)"
        echo "v2:         ${v2_time}s (${v2_speedup}x)" 
        echo "v2_simple:  ${v2s_time}s (${v2s_speedup}x)"
        # echo "v2_swap:    ${v2p_time}s (${v2p_speedup}x)"
        echo ""
    done
    echo ""
done