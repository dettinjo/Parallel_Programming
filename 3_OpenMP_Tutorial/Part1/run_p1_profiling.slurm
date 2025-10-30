#!/bin/bash

# =======================================================================
# SLURM DIRECTIVES FOR PART 1
# =======================================================================
# -- Set the job name for easy identification
#SBATCH --job-name=P1_Seq_Profile

# -- Set the file for standard output (%j will be replaced by the job ID)
#SBATCH --output=p1_slurm_log_%j.out

# -- Set the file for standard error
#SBATCH --error=p1_slurm_log_%j.err

# -- Request a single task for our sequential program
#SBATCH --ntasks=1

# -- Request 1 CPU core for the task
#SBATCH --cpus-per-task=1

# -- Set a time limit (e.g., 5 minutes). The job will be killed if it exceeds this.
#SBATCH --time=00:05:00

# -- Request memory for the job
#SBATCH --mem-per-cpu=2G

# -- Set partition for executing the job
#SBATCH --partition=cuda-ext.q

# =======================================================================
# EXECUTION LOGIC FOR PART 1
# =======================================================================

# -- Define filenames for clarity --
SOURCE_FILE="Prg.c"
EXEC_NAME="Prg"

# -- Start Message --
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running Part 1 (P1) Sequential Profiling on node: $(hostname)"
echo "----------------------------------------------------"

# -- Step 1: Compile the sequential code --
echo "Compiling $SOURCE_FILE with -Ofast and -fno-inline..."
gcc -Ofast -fno-inline "$SOURCE_FILE" -o "$EXEC_NAME" -lm

# Check for compilation success
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi
echo "Compilation successful."
echo "----------------------------------------------------"

# -- Step 2: Run the program with perf record --
echo "Running 'perf record' on the executable..."
# This command will run the program and generate a 'perf.data' file
perf record ./"$EXEC_NAME"

echo "----------------------------------------------------"
echo "Job finished. The 'perf.data' file has been created."
echo "To analyze the results, run 'perf report' in your terminal after the job completes."