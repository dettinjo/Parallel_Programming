#!/bin/bash

# =======================================================================
# SLURM DIRECTIVES FOR PART 3
# =======================================================================
#SBATCH --job-name=P3_Correctness_Check
#SBATCH --output=04_p3_slurm_log_%j.out
#SBATCH --error=06_p3_slurm_log_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=cuda-ext.q
#SBATCH --nodelist=aolin-gpu-1

# =======================================================================
# EXECUTION LOGIC FOR PART 3
# =======================================================================

# -- Define filenames and parameters --
SOURCE_FILE="02_PrgPAR2.c"
EXEC_NAME="03_PrgPAR2"
OUTPUT_FILE="05_p3_output_%j.txt"
NUM_THREADS=2

# -- Start Message --
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running Part 3 (P3) Correctness Check on node: $(hostname)"
echo "----------------------------------------------------"

# -- Step 1: Compile the OpenMP code --
echo "Compiling $SOURCE_FILE with -fopenmp..."
gcc -Ofast -fno-inline -fopenmp "$SOURCE_FILE" -o "$EXEC_NAME" -lm

# Check for compilation success
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi
echo "Compilation successful."
echo "----------------------------------------------------"

# -- Step 2: Set the number of threads for the execution --
export OMP_NUM_THREADS=$NUM_THREADS
echo "OMP_NUM_THREADS set to $NUM_THREADS"

# -- Step 3: Run the program with 'time' and save all output --
echo "Running 'time ./${EXEC_NAME}' and saving output to $OUTPUT_FILE..."

# The syntax { ... ; } 2>&1 > "$OUTPUT_FILE" is a robust way to ensure
# that BOTH the standard output of the program AND the standard error output
# from the 'time' command are redirected into the same file.
{ time ./"$EXEC_NAME" ; } 2>&1 > "$OUTPUT_FILE"

echo "----------------------------------------------------"
echo "Job finished. The complete execution and timing output is in '$OUTPUT_FILE'."