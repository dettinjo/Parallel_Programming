#!/bin/bash

#=======================================================================
# SLURM DIRECTIVES
#=======================================================================

# --Job name
#SBATCH --job-name=PP_Laplace_Tests

# --Output file for the script's own messages (%j will be the job ID)
#SBATCH --output=slurm_log_%j.out

# --Error file for the script's own errors
#SBATCH --error=slurm_log_%j.err

# --Number of tasks
#SBATCH --ntasks=1

# --Request 1 CPU core for our task
#SBATCH --cpus-per-task=1

# --Increased time limit for running all tests (30 minutes)
#SBATCH --time=00:30:00

# --Increased memory for the largest job (4GB)
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=cuda-ext.q


#=======================================================================
# SCRIPT SETUP
#=======================================================================

# Define a single output file for all results
OUTPUT_FILE="laplace_report.txt"

# --- NEW: Initialize variables to store execution times ---
# This is good practice in case a run fails.
time_static_100="FAILED"
time_static_1000="FAILED"
time_static_4096="FAILED"
time_dyn_100="FAILED"
time_dyn_1000="FAILED"
time_dyn_4096="FAILED"
time_opt_100="FAILED"
time_opt_1000="FAILED"
time_opt_4096="FAILED"

#=======================================================================
# EXECUTION LOGIC - PART 1: RUNNING JOBS & CAPTURING DATA
#=======================================================================

# Create a clean output file and add a header.
echo "=================================================" > $OUTPUT_FILE
echo "   Jacobi Solver - Full Execution Logs" >> $OUTPUT_FILE
echo "   Job started at: $(date)" >> $OUTPUT_FILE
echo "=================================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# --- Static Versions ---
echo "--- LOG: Running Static Version (100x100) ---" >> $OUTPUT_FILE
(time ./stat_100) >> $OUTPUT_FILE 2>&1
# --- NEW: Capture the time into a variable ---
time_static_100=$({ time ./stat_100; } 2>&1 | grep real | awk '{print $2}')
echo -e "\n\n" >> $OUTPUT_FILE

echo "--- LOG: Running Static Version (1000x1000) ---" >> $OUTPUT_FILE
(time ./stat_1000) >> $OUTPUT_FILE 2>&1
time_static_1000=$({ time ./stat_1000; } 2>&1 | grep real | awk '{print $2}')
echo -e "\n\n" >> $OUTPUT_FILE

echo "--- LOG: Running Static Version (4096x4096) ---" >> $OUTPUT_FILE
(time ./stat_4096) >> $OUTPUT_FILE 2>&1
time_static_4096=$({ time ./stat_4096; } 2>&1 | grep real | awk '{print $2}')
echo -e "\n\n" >> $OUTPUT_FILE


# --- Dynamic and Optimized Versions (using a loop) ---
for size in 100 1000 4096
do
    # Run the dynamic version
    echo "--- LOG: Running Dynamic Version ($size x $size) ---" >> $OUTPUT_FILE
    (time ./dyn $size $size) >> $OUTPUT_FILE 2>&1
    # --- NEW: Capture time into the correct variable based on size ---
    if [ $size -eq 100 ]; then
        time_dyn_100=$({ time ./dyn $size $size; } 2>&1 | grep real | awk '{print $2}')
    elif [ $size -eq 1000 ]; then
        time_dyn_1000=$({ time ./dyn $size $size; } 2>&1 | grep real | awk '{print $2}')
    elif [ $size -eq 4096 ]; then
        time_dyn_4096=$({ time ./dyn $size $size; } 2>&1 | grep real | awk '{print $2}')
    fi
    echo -e "\n\n" >> $OUTPUT_FILE

    # Run the optimized version
    echo "--- LOG: Running Optimized Version ($size x $size) ---" >> $OUTPUT_FILE
    (time ./opt $size $size) >> $OUTPUT_FILE 2>&1
    # --- NEW: Capture time into the correct variable based on size ---
    if [ $size -eq 100 ]; then
        time_opt_100=$({ time ./opt $size $size; } 2>&1 | grep real | awk '{print $2}')
    elif [ $size -eq 1000 ]; then
        time_opt_1000=$({ time ./opt $size $size; } 2>&1 | grep real | awk '{print $2}')
    elif [ $size -eq 4096 ]; then
        time_opt_4096=$({ time ./opt $size $size; } 2>&1 | grep real | awk '{print $2}')
    fi
    echo -e "\n\n" >> $OUTPUT_FILE
done


#=======================================================================
# EXECUTION LOGIC - PART 2: APPENDING THE SUMMARY TABLE
#=======================================================================

# --- NEW: This entire section appends the final summary table ---
echo "=================================================" >> $OUTPUT_FILE
echo "            PERFORMANCE SUMMARY" >> $OUTPUT_FILE
echo "=================================================" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
printf "%-12s | %-20s | %-20s | %-20s\n" "Grid Size" "V1 (Static)" "V2 (Dynamic)" "V3 (Optimized)" >> $OUTPUT_FILE
printf "%-12s | %-20s | %-20s | %-20s\n" "------------" "--------------------" "--------------------" "--------------------" >> $OUTPUT_FILE
printf "%-12s | %-20s | %-20s | %-20s\n" "100x100" "$time_static_100" "$time_dyn_100" "$time_opt_100" >> $OUTPUT_FILE
printf "%-12s | %-20s | %-20s | %-20s\n" "1000x1000" "$time_static_1000" "$time_dyn_1000" "$time_opt_1000" >> $OUTPUT_FILE
printf "%-12s | %-20s | %-20s | %-20s\n" "4096x4096" "$time_static_4096" "$time_dyn_4096" "$time_opt_4096" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "Job finished at: $(date)" >> $OUTPUT_FILE
echo "=================================================" >> $OUTPUT_FILE

echo "Job finished. Full report and summary table are in $OUTPUT_FILE"