#!/bin/bash

# =======================================================================
# SLURM DIRECTIVES FOR PART 1 (Corrected & Combined)
# =======================================================================
#SBATCH --job-name=P1_Seq_Profile
#SBATCH --output=05_p1_slurm_log_%j.out
#SBATCH --error=06_p1_slurm_log_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=cuda-ext.q
#SBATCH --nodelist=aolin-gpu-1

# =======================================================================
# EXECUTION LOGIC FOR PART 1 (Corrected & Combined)
# =======================================================================

# -- Define filenames --
# We'll use the original Prg.c, which you should have
SOURCE_FILE="02_Prg.c"
EXEC_NAME="03_Prg_P1"

# -- Define explicit arguments for consistency --
N_VAL=20000
REP_VAL=250000

# -- Start Message --
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running Part 1 (P1) Sequential Timing & Profiling on node: $(hostname)"
echo "Using $SOURCE_FILE with N=$N_VAL REP=$REP_VAL"
echo "----------------------------------------------------"

# -- Step 1: Compile the sequential code --
echo "Compiling $SOURCE_FILE with -Ofast and -fno-inline..."
# -fno-inline is critical for perf to see the separate functions
# -lm is for the math library (fabs, trunc)
gcc -Ofast -fno-inline "$SOURCE_FILE" -o "$EXEC_NAME" -lm

# Check for compilation success
if [ $? -ne 0 ]; then
    echo "Compilation failed! Did you forget to upload $SOURCE_FILE?"
    exit 1
fi
echo "Compilation successful."
echo "----------------------------------------------------"

# -- Step 2: Run 'perf stat' for total time (Task 1 of P1) --
echo "Running 'perf stat' for overall timing..."
# This will show the ~49.6 second runtime
perf stat ./"$EXEC_NAME" $N_VAL $REP_VAL

echo "----------------------------------------------------"

# -- Step 3: Run 'perf record' for profiling (Task 2 of P1) --
echo "Running 'perf record' to generate profiling data..."
# This will run the program and generate a 'perf.data' file
perf record ./"$EXEC_NAME" $N_VAL $REP_VAL

echo "----------------------------------------------------"
echo "Job finished."
echo "The 'perf stat' output above shows the true sequential time (~49.6s)."
echo ""
echo "A 'perf.data' file has been created."
echo "To analyze the profiling results (like in the PDF), run:"
echo "perf report"
echo "----------------------------------------------------"

