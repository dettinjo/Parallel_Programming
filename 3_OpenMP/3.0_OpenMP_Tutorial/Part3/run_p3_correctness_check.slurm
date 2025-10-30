#!/bin/bash

# =======================================================================
# SLURM DIRECTIVES FOR PART 3
# =======================================================================
# -- Set the job name for easy identification
#SBATCH --job-name=P3_Correctness_Check

# -- Set the file for standard output (%j will be replaced by the job ID)
#SBATCH --output=p3_slurm_log_%j.out

# -- Set the file for standard error
#SBATCH --error=p3_slurm_log_%j.err

# -- Request a single task (the program itself manages threads)
#SBATCH --ntasks=1

# -- Request 2 CPU cores for the 2 threads
#SBATCH --cpus-per-task=2

# -- Set a time limit (e.g., 2 minutes is sufficient)
#SBATCH --time=00:02:00

# -- Request memory for the job
#SBATCH --mem-per-cpu=2G

# -- Set partition for executing the job
#SBATCH --partition=cuda-ext.q

# =======================================================================
# EXECUTION LOGIC FOR PART 3
# =======================================================================

# -- Define filenames and parameters --
SOURCE_FILE="PrgPAR2.c"
EXEC_NAME="PrgPAR2"
OUTPUT_FILE="p3_execution_output.txt"
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