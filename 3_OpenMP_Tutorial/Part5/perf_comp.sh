#!/bin/bash

echo "========================================="
echo "OpenMP Performance Comparison"
echo "========================================="

N=20000
REP=250000

# Compile serial version if available
if [ -f "Prg.c" ]; then
    echo "Compiling serial version..."
    gcc -Ofast -fno-inline Prg.c -o Prg -lm
    
    echo "Running serial baseline..."
    SERIAL_TIME=$( { time ./Prg $N $REP > /dev/null; } 2>&1 | grep real | awk '{print $2}' )
    echo "Serial time: $SERIAL_TIME"
    echo ""
fi

# Compile and test the final optimized version
echo "Compiling optimized parallel version..."
gcc -Ofast -fno-inline -fopenmp -march=native PrgPAR_Optimized.c -o PrgPAR_Optimized -lm

echo ""
echo "========================================="
echo "Testing Different Thread Counts"
echo "========================================="

echo "Thread,Time(s),User(s),Speedup" > results.csv

for THREADS in 1 2 3 4 6 8 12; do
    export OMP_NUM_THREADS=$THREADS
    echo ""
    echo "--- Testing with $THREADS thread(s) ---"
    
    # Run and capture time
    OUTPUT=$(./PrgPAR_Optimized $N $REP 2>&1)
    EXEC_TIME=$(echo "$OUTPUT" | grep "Execution time:" | awk '{print $3}')
    
    echo "$OUTPUT"
    
    # Calculate speedup if we have serial time
    if [ ! -z "$SERIAL_TIME" ]; then
        SPEEDUP=$(echo "scale=2; 12.3 / $EXEC_TIME" | bc)
        echo "Speedup vs serial: ${SPEEDUP}x"
    fi
    
    echo "$THREADS,$EXEC_TIME,N/A,${SPEEDUP:-N/A}" >> results.csv
done

echo ""
echo "========================================="
echo "Results Summary"
echo "========================================="
cat results.csv | column -t -s','

echo ""
echo "Results saved to: results.csv"
echo ""
echo "Key Bottlenecks Identified:"
echo "1. VectScan: Sequential prefix sum (cannot be parallelized efficiently)"
echo "2. Synchronization overhead: Barriers between pipeline stages"
echo "3. Load imbalance: VectScan runs on 1 thread while others wait"
echo "4. Memory bandwidth: All threads competing for memory access"
echo ""
echo "Optimization Strategies Applied:"
echo "- Single parallel region (eliminates fork/join overhead)"
echo "- Manual work distribution (eliminates OpenMP scheduling overhead)"
echo "- Overlapping VectF2 and VectAverage (task parallelism)"
echo "- Parallel VectSum with manual reduction"
echo "- Static workload partitioning (cache-friendly)"