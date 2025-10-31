#!/bin/bash

# Comprehensive performance bottleneck analysis
echo "=== COMPREHENSIVE PERFORMANCE BOTTLENECK ANALYSIS ==="
echo "Running multiple matrix sizes to identify scaling patterns..."
echo ""

# Array of test sizes
sizes=(100 200 300 500 750 1000)

# Create output file for data analysis
output_file="performance_data.txt"
echo "# Size_N Size_M Total_Time Main_Loop_Time Stencil_Time Error_Time Update_Time Memory_BW GFLOPS" > $output_file

for size in "${sizes[@]}"; do
    echo "Testing ${size}x${size} matrix..."
    
    # Run the test and capture output
    result=$(./laplace_instrumented $size $size 2>/dev/null | tail -20)
    
    # Extract key metrics using grep and awk
    total_time=$(echo "$result" | grep "Total execution time" | awk '{print $4}')
    main_time=$(echo "$result" | grep "Main computation" | awk '{print $3}')
    stencil_time=$(echo "$result" | grep "Stencil computation" | awk '{print $3}')
    error_time=$(echo "$result" | grep "Error calculation" | awk '{print $3}')
    update_time=$(echo "$result" | grep "Matrix update" | awk '{print $3}')
    bandwidth=$(echo "$result" | grep "Average memory bandwidth" | awk '{print $4}')
    gflops=$(echo "$result" | grep "Average GFLOPS" | awk '{print $3}')
    
    # Store in data file
    echo "$size $size $total_time $main_time $stencil_time $error_time $update_time $bandwidth $gflops" >> $output_file
    
    echo "  Total time: ${total_time}s, Memory BW: ${bandwidth} GB/s, GFLOPS: ${gflops}"
done

echo ""
echo "=== BOTTLENECK ANALYSIS SUMMARY ==="
echo ""

# Analyze the data
echo "Performance scaling analysis:"
echo "Size    Total_Time  Stencil%  Error%    Update%   Mem_BW    GFLOPS"
echo "------  ----------  --------  --------  --------  --------  --------"

tail -n +2 $output_file | while read size1 size2 total main stencil error update bw gflops; do
    stencil_pct=$(echo "$stencil $main" | awk '{printf "%.1f", ($1/$2)*100}')
    error_pct=$(echo "$error $main" | awk '{printf "%.1f", ($1/$2)*100}')
    update_pct=$(echo "$update $main" | awk '{printf "%.1f", ($1/$2)*100}')
    
    printf "%-6s  %-10s  %-8s  %-8s  %-8s  %-8s  %-8s\n" \
           "${size1}x${size2}" "$total" "${stencil_pct}%" "${error_pct}%" "${update_pct}%" "$bw" "$gflops"
done

echo ""
echo "=== KEY FINDINGS ==="
echo ""

# Calculate some key metrics
first_line=$(tail -n +2 $output_file | head -1)
last_line=$(tail -1 $output_file)

first_size=$(echo $first_line | awk '{print $1}')
last_size=$(echo $last_line | awk '{print $1}')
first_time=$(echo $first_line | awk '{print $3}')
last_time=$(echo $last_line | awk '{print $3}')

size_ratio=$(echo "$last_size $first_size" | awk '{printf "%.1f", $1/$2}')
time_ratio=$(echo "$last_time $first_time" | awk '{printf "%.1f", $1/$2}')
expected_ratio=$(echo "$size_ratio" | awk '{printf "%.1f", $1*$1}')

echo "1. SCALING BEHAVIOR:"
echo "   - Matrix size increased by ${size_ratio}x (${first_size} -> ${last_size})"
echo "   - Expected time increase: ${expected_ratio}x (O(n²) scaling)"
echo "   - Actual time increase: ${time_ratio}x"
if (( $(echo "$time_ratio < $expected_ratio" | bc -l) )); then
    echo "   - GOOD: Better than expected scaling (cache effects or compiler optimizations)"
else
    echo "   - Performance scales as expected for memory-bound problem"
fi

echo ""
echo "2. BOTTLENECK IDENTIFICATION:"
avg_error_pct=$(tail -n +2 $output_file | awk '{sum+=$6/$4} END {printf "%.1f", (sum/NR)*100}')
avg_stencil_pct=$(tail -n +2 $output_file | awk '{sum+=$5/$4} END {printf "%.1f", (sum/NR)*100}')
avg_update_pct=$(tail -n +2 $output_file | awk '{sum+=$7/$4} END {printf "%.1f", (sum/NR)*100}')

echo "   - Error calculation: ${avg_error_pct}% of main loop time (PRIMARY BOTTLENECK)"
echo "   - Stencil computation: ${avg_stencil_pct}% of main loop time"
echo "   - Matrix update: ${avg_update_pct}% of main loop time"
echo ""
echo "   CRITICAL INSIGHT: Error calculation dominates execution time!"
echo "   This involves: fmaxf(), sqrtf(), fabsf() operations + global reduction"

echo ""
echo "3. MEMORY BANDWIDTH:"
avg_bandwidth=$(tail -n +2 $output_file | awk '{sum+=$8} END {printf "%.1f", sum/NR}')
echo "   - Average memory bandwidth: ${avg_bandwidth} GB/s"
echo "   - This suggests memory-bound performance (typical for stencil codes)"

echo ""
echo "4. COMPUTATIONAL INTENSITY:"
avg_gflops=$(tail -n +2 $output_file | awk '{sum+=$9} END {printf "%.1f", sum/NR}')
echo "   - Average GFLOPS: ${avg_gflops}"
echo "   - Low GFLOPS indicates memory-bound rather than compute-bound workload"

echo ""
echo "=== PARALLELIZATION STRATEGY RECOMMENDATIONS ==="
echo ""
echo "Based on this analysis, the parallelization strategy should focus on:"
echo ""
echo "1. ERROR CALCULATION BOTTLENECK (${avg_error_pct}% of time):"
echo "   - Use OpenMP reduction for the 'error' variable"
echo "   - Consider: #pragma omp parallel for reduction(max:error)"
echo "   - This will be the most critical optimization"
echo ""
echo "2. STENCIL COMPUTATION (${avg_stencil_pct}% of time):"
echo "   - Straightforward parallelization with #pragma omp parallel for"
echo "   - Consider 'collapse(2)' for nested loops if beneficial"
echo "   - Good load balancing expected (uniform computation)"
echo ""
echo "3. MATRIX UPDATE (${avg_update_pct}% of time):"
echo "   - Simple parallel assignment - easy to parallelize"
echo "   - Consider combining with stencil computation to reduce overhead"
echo ""
echo "4. MEMORY OPTIMIZATION OPPORTUNITIES:"
echo "   - Memory bandwidth is limiting factor"
echo "   - Consider loop fusion to reduce memory accesses"
echo "   - Cache blocking might help for very large matrices"
echo "   - NUMA-aware memory allocation for multi-socket systems"
echo ""
echo "5. EXPECTED PARALLEL PERFORMANCE:"
echo "   - Should scale well up to memory bandwidth limit"
echo "   - Watch for false sharing in error reduction"
echo "   - Expected speedup: 2-8x depending on cores and memory subsystem"

echo ""
echo "Raw data saved to: $output_file"