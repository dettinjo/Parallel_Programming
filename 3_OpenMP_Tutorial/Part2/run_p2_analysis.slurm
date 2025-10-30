#!/bin/bash

# =======================================================================
# SLURM DIRECTIVES FOR PART 2
# =======================================================================
# -- Set the job name
#SBATCH --job-name=P2_Flawed_OMP

# -- Set the file for standard output
#SBATCH --output=p2_slurm_log_%j.out

# -- Set the file for standard error
#SBATCH --error=p2_slurm_log_%j.err

# -- Request a single task (the program itself manages threads)
#SBATCH --ntasks=1

# -- Request 2 CPU cores for the 2 threads
#SBATCH --cpus-per-task=2

# -- Set a time limit (e.g., 2 minutes)
#SBATCH --time=00:02:00

# -- Request memory for the job
#SBATCH --mem-per-cpu=2G

# -- Set partition for executing the job
#SBATCH --partition=cuda-ext.q

#SBATCH --nodelist=aolin-gpu-1

# =======================================================================
# EXECUTION LOGIC FOR PART 2
# =======================================================================

# -- Define filenames and parameters --
SOURCE_FILE="PrgPAR.c"
EXEC_NAME="PrgPAR"
OUTPUT_FILE="p2_execution_output.txt"
NUM_THREADS=2

# -- Start Message --
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running Part 2 (P2) Flawed Parallel Analysis on node: $(hostname)"
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
echo "OMP_NUM_THREADS set to $OMP_NUM_THREADS"

# -- Step 3: Run the program and save its output to a file --
echo "Running the program and saving output to $OUTPUT_FILE..."
# The single '>' creates the output file and writes the program's output into it
./"$EXEC_NAME" > "$OUTPUT_FILE"

echo "----------------------------------------------------"
echo "Job finished. The complete execution output is in '$OUTPUT_FILE'."