#!/bin/bash
#SBATCH --job-name=laplace-benchmark
#SBATCH --output=laplace-benchmark-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=research.q
#SBATCH --exclusive

# Compile all versions
echo "Compiling versions..."
gcc -Wall -O3 laplace-static.c -o laplace-static -lm
gcc -Wall -O3 laplace-dynamic.c -o laplace-dynamic -lm
gcc -Wall -O3 laplace-optimized.c -o laplace-optimized -lm
echo "Compilation complete"
echo ""

# Test parameters
SIZES=(100 1000 4096)
ITERATIONS=(100 1000 2000)
TOLERANCE=1e-9
RESULTS_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

# Initialize results file
{
    echo "Laplace Solver Benchmark Results"
    echo "Date: $(date)"
    echo "Host: $(hostname)"
    echo "Tolerance: $TOLERANCE"
    echo ""
    printf "%-20s %-12s %-12s %-15s\n" "Version" "Size" "Iterations" "Time (sec)"
    printf "%s\n" "----------------------------------------------------------------"
} > "$RESULTS_FILE"

# Timing function
measure_time() {
    local start=$(date +%s.%N)
    "$@" > /dev/null 2>&1
    local end=$(date +%s.%N)
    echo "$end - $start" | bc
}

echo "Starting benchmark on $(hostname)"
echo ""

# Benchmark laplace-static
echo "Testing laplace-static..."
for ITER in "${ITERATIONS[@]}"; do
    TIME=$(measure_time ./laplace-static $ITER $TOLERANCE)
    printf "  15x15, iter=%5d: %10.6f sec\n" "$ITER" "$TIME"
    printf "%-20s %-12s %-12s %-15s\n" "laplace-static" "15x15" "$ITER" "$TIME" >> "$RESULTS_FILE"
done
echo ""

# Benchmark laplace-dynamic
echo "Testing laplace-dynamic..."
for SIZE in "${SIZES[@]}"; do
    for ITER in "${ITERATIONS[@]}"; do
        # Skip 4096x4096 with 10000 iterations (too slow)
        if [ $SIZE -eq 4096 ] && [ $ITER -eq 10000 ]; then
            echo "  ${SIZE}x${SIZE}, iter=${ITER}: SKIPPED (too slow)"
            printf "%-20s %-12s %-12s %-15s\n" "laplace-dynamic" "${SIZE}x${SIZE}" "$ITER" "SKIPPED" >> "$RESULTS_FILE"
            continue
        fi
        
        TIME=$(measure_time ./laplace-dynamic $SIZE $SIZE $ITER $TOLERANCE)
        printf "  %dx%d, iter=%5d: %10.6f sec\n" "$SIZE" "$SIZE" "$ITER" "$TIME"
        printf "%-20s %-12s %-12s %-15s\n" "laplace-dynamic" "${SIZE}x${SIZE}" "$ITER" "$TIME" >> "$RESULTS_FILE"
    done
done
echo ""

# Benchmark laplace-optimized
echo "Testing laplace-optimized..."
for SIZE in "${SIZES[@]}"; do
    for ITER in "${ITERATIONS[@]}"; do
        # Skip 4096x4096 with 10000 iterations (too slow)
        if [ $SIZE -eq 4096 ] && [ $ITER -eq 10000 ]; then
            echo "  ${SIZE}x${SIZE}, iter=${ITER}: SKIPPED (too slow)"
            printf "%-20s %-12s %-12s %-15s\n" "laplace-optimized" "${SIZE}x${SIZE}" "$ITER" "SKIPPED" >> "$RESULTS_FILE"
            continue
        fi
        
        TIME=$(measure_time ./laplace-optimized $SIZE $SIZE $ITER $TOLERANCE)
        printf "  %dx%d, iter=%5d: %10.6f sec\n" "$SIZE" "$SIZE" "$ITER" "$TIME"
        printf "%-20s %-12s %-12s %-15s\n" "laplace-optimized" "${SIZE}x${SIZE}" "$ITER" "$TIME" >> "$RESULTS_FILE"
    done
done

echo ""
echo "Benchmark complete. Results saved to: $RESULTS_FILE"
echo ""
cat "$RESULTS_FILE"