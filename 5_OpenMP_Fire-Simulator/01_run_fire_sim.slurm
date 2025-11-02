#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --partition=cuda-ext.q
#SBATCH --output=11_fire_sim_out_%j.out
#SBATCH --error=12_fire_sim_err_%j.err
#SBATCH --time=00:30:00
#SBATCH --job-name=OpenMP_FireSim
#SBATCH --nodelist=aolin-gpu-1

# 1. LOAD MODULES
echo "--- Loading modules ---" >&2
# Load the modern GCC compiler
module add gcc/12.2.1
echo "--- Modules loaded ---" >&2
echo "" >&2

# 2. COMPILE EXECUTABLES
echo "--- Compiling executables ---" >&2
# -O3: High optimization level
# -march=native: Optimizes for the *specific CPU* of this node (aolin-gpu-1)
# -Wall -Wextra: Show all good warnings
gcc -O3 -march=native -Wall -Wextra -o 05_extinguishing_sequential 02_extinguishing_sequential.c -lm
gcc -O3 -march=native -Wall -Wextra -fopenmp -o 06_extinguishing_openmp 03_extinguishing_openmp.c -lm
echo "--- Compilation complete ---" >&2
echo "" >&2

# 3. DEFINE TEST PARAMETERS
TEST_FILES=("test_files/test1" "test_files/test2" "test_files/test3" "test_files/test4")
THREADS=(6 12 24)

# Storage for CSV data
CSV_SPEEDUP="TestFile,Version,NumThreads,BaselineTime_s,ParallelTime_s,Speedup_x,Efficiency_pc\n"
CSV_BOTTLENECK="TestFile,Version,NumThreads,Time_Focal_s,Time_Heat_s,Time_Move_s,Time_Action_s\n"

# 4. ANALYSIS SCRIPT - HUMAN-READABLE TABLE
echo "=========================================================================================================="
echo "                           PART 1: SPEEDUP & EFFICIENCY ANALYSIS                                        "
echo "=========================================================================================================="
printf "%-18s | %-20s | %-8s | %-17s | %-17s | %-12s\n" "Test Case" "Version" "Threads" "Baseline Time (s)" "Parallel Time (s)" "Speedup (x)"
echo "-------------------+----------------------+----------+-------------------+-------------------+--------------"

# Create a second table for bottleneck analysis
TABLE_BOTTLENECK="\n\n"
TABLE_BOTTLENECK+="=========================================================================================================\n"
TABLE_BOTTLENECK+="                           PART 2: BOTTLENECK ANALYSIS (Time per section)                             \n"
TABLE_BOTTLENECK+="=========================================================================================================\n"
TABLE_BOTTLENECK+=$(printf "%-18s | %-20s | %-8s | %-12s | %-12s | %-12s | %-12s\n" "Test Case" "Version" "Threads" "Focal (s)" "Heat (s)" "Move (s)" "Action (s)")
TABLE_BOTTLENECK+="\n-------------------+----------------------+----------+--------------+--------------+--------------+--------------\n"


# Function to parse the output using a single awk command
parse_output() {
    echo "$1" | awk '
        /Time:/ { time = $2 }
        /Stats_Focal:/ { focal = $2 }
        /Stats_Heat:/ { heat = $2 }
        /Stats_Move:/ { move = $2 }
        /Stats_Action:/ { action = $2 }
        END { print time, focal, heat, move, action }
    '
}

# Iterate over each test file
for FILE in "${TEST_FILES[@]}"; do
    echo "Running test: $FILE" >&2 # Print progress to stderr

    # Run sequential version once to get baseline time and stats
    OUTPUT_SEQ=$(./05_extinguishing_sequential -f $FILE)
    read TIME_SEQ STATS_FOCAL_SEQ STATS_HEAT_SEQ STATS_MOVE_SEQ STATS_ACTION_SEQ < <(parse_output "$OUTPUT_SEQ")
    
    if [ -z "$TIME_SEQ" ]; then
        echo "Error running sequential baseline for $FILE" >&2
        printf "%-18s | %-20s | %-8s | %-17s | %-17s | %-12s\n" $FILE "Sequential" "1" "ERROR" "N/A" "N/A"
        continue
    fi
    
    # Print baseline data to Speedup table
    printf "%-18s | %-20s | %-8s | %-17s | %-17s | %-12s\n" $FILE "Sequential" "1" $TIME_SEQ $TIME_SEQ "1.0000"
    CSV_SPEEDUP+="$FILE,Sequential,1,$TIME_SEQ,$TIME_SEQ,1.0000,100.00\n"
    
    # Print baseline data to Bottleneck table
    TABLE_BOTTLENECK+=$(printf "%-18s | %-20s | %-8s | %-12s | %-12s | %-12s | %-12s\n" $FILE "Sequential" "1" $STATS_FOCAL_SEQ $STATS_HEAT_SEQ $STATS_MOVE_SEQ $STATS_ACTION_SEQ)
    TABLE_BOTTLENECK+="\n"
    CSV_BOTTLENECK+="$FILE,Sequential,1,$STATS_FOCAL_SEQ,$STATS_HEAT_SEQ,$STATS_MOVE_SEQ,$STATS_ACTION_SEQ\n"

    
    # --- Test OMP (Improved) ---
    echo "  Testing OMP (Improved)..." >&2
    for T in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$T
        OUTPUT_OMP=$(./06_extinguishing_openmp -f $FILE)
        read TIME_OMP STATS_FOCAL STATS_HEAT STATS_MOVE STATS_ACTION < <(parse_output "$OUTPUT_OMP")
        
        # Check if TIME_OMP is non-empty and greater than 0
        if [ ! -z "$TIME_OMP" ] && [ $(echo "$TIME_OMP > 0" | bc) -eq 1 ]; then
            SPEEDUP=$(echo "scale=4; $TIME_SEQ / $TIME_OMP" | bc)
            EFFICIENCY=$(echo "scale=2; ($SPEEDUP / $T) * 100" | bc)
            
            printf "%-18s | %-20s | %-8s | %-17s | %-17s | %-12s\n" $FILE "OMP_Improved" $T $TIME_SEQ $TIME_OMP $SPEEDUP
            CSV_SPEEDUP+="$FILE,OMP_Improved,$T,$TIME_SEQ,$TIME_OMP,$SPEEDUP,$EFFICIENCY\n"
            
            TABLE_BOTTLENECK+=$(printf "%-18s | %-20s | %-8s | %-12s | %-12s | %-12s | %-12s\n" $FILE "OMP_Improved" $T $STATS_FOCAL $STATS_HEAT $STATS_MOVE $STATS_ACTION)
            TABLE_BOTTLENECK+="\n"
            CSV_BOTTLENECK+="$FILE,OMP_Improved,$T,$STATS_FOCAL,$STATS_HEAT,$STATS_MOVE,$STATS_ACTION\n"
        else
            # Handle error case
            printf "%-18s | %-20s | %-8s | %-17s | %-17s | %-12s\n" $FILE "OMP_Improved" $T $TIME_SEQ "ERROR" "N/A"
            CSV_SPEEDUP+="$FILE,OMP_Improved,$T,$TIME_SEQ,ERROR,N/A,N/A\n"
            TABLE_BOTTLENECK+=$(printf "%-18s | %-20s | %-8s | %-12s | %-12s | %-12s | %-12s\n" $FILE "OMP_Improved" $T "ERR" "ERR" "ERR" "ERR")
            TABLE_BOTTLENECK+="\n"
            CSV_BOTTLENECK+="$FILE,OMP_Improved,$T,ERROR,ERROR,ERROR,ERROR\n"
        fi
    done
    echo "-------------------+----------------------+----------+-------------------+-------------------+--------------"
done

# Print the bottleneck table
echo -e $TABLE_BOTTLENECK
echo ""
echo ""

# 5. PERF STAT ANALYSIS
echo "====================================================================================="
echo "                PERF STAT ANALYSIS (test_files/test4 - The Bottleneck)               "
echo "====================================================================================="
echo "--- Running perf stat on SEQUENTIAL (test_files/test4) ---" >&2
perf stat -e cycles,instructions,cache-references,cache-misses,task-clock,L1-dcache-load-misses ./05_extinguishing_sequential -f test_files/test4
echo ""
echo "--- Running perf stat on OMP_Improved (24 Threads) ---" >&2
export OMP_NUM_THREADS=24
perf stat -e cycles,instructions,cache-references,cache-misses,task-clock,L1-dcache-load-misses ./06_extinguishing_openmp -f test_files/test4
echo "--- Analysis Complete ---"
echo ""
echo ""


# 6. SAVE CSV DATA TO FILES
echo "====================================================================================="
echo "                         SAVING CSV DATA TO FILES                                    "
echo "====================================================================================="
# Save the CSV data from the bash variables into files
echo -e "$CSV_SPEEDUP" > 07_speedup_data.csv
echo -e "$CSV_BOTTLENECK" > 08_bottleneck_data.csv
echo "... saved 07_speedup_data.csv and 08_bottleneck_data.csv"
echo ""
# 7. JOB COMPLETE
echo "====================================================================================="
echo "                         SLURM JOB COMPLETE                                          "
echo "====================================================================================="