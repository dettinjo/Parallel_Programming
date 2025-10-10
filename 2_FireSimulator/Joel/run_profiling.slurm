#!/bin/bash

# =======================================================================
# SLURM DIRECTIVES (Instructions for the Slurm Job Manager)
# =======================================================================

# -- Set the job name
#SBATCH --job-name=FireSimProfile

# -- Set the file to redirect standard output to (%j = Job ID)
#SBATCH --output=fire_sim_log_%j.out

# -- Set the file to redirect standard error to
#SBATCH --error=fire_sim_log_%j.err

# -- Request a single task (since this is a sequential program)
#SBATCH --ntasks=1

# -- Request 1 CPU core for the task
#SBATCH --cpus-per-task=1

# -- Set a time limit for the job
#SBATCH --time=00:30:00

# -- Request memory for the job (2GB is safe for the largest test case)
#SBATCH --mem-per-cpu=2G

# -- Set the partition for the job
#SBATCH --partition=cuda-ext.q


# =======================================================================
# SCRIPT SETUP (Defining variables for clarity)
# =======================================================================

# The C source file to be compiled
SOURCE_FILE="extinguishing_timed.c"

# The name of the executable that will be created
EXEC_NAME="extinguishing_timed"

# The single output file where all results will be saved
OUTPUT_FILE="profiling_results.txt"

# An array of the test cases to run
TEST_CASES=("test2" "test3" "test4")


# =======================================================================
# EXECUTION LOGIC
# =======================================================================

# -- Step 1: Print a start message to the Slurm log --
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Starting job on node: $(hostname)"
echo "----------------------------------------------------"


# -- Step 2: Compile the C code --
echo "Compiling source file: $SOURCE_FILE..."
gcc -Ofast "$SOURCE_FILE" -o "$EXEC_NAME" -lm

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed! Exiting."
    exit 1
fi
echo "Compilation successful. Executable name: $EXEC_NAME"
echo "----------------------------------------------------"


# -- Step 3: Initialize the results file --
# The single '>' operator creates a new file (or overwrites an old one)
# and adds a header.
echo "Fire Simulation Profiling Results" > "$OUTPUT_FILE"
echo "Job executed on: $(hostname) at $(date)" >> "$OUTPUT_FILE"
echo "=================================================" >> "$OUTPUT_FILE"


# -- Step 4: Loop through the test cases and run the program --
echo "Starting benchmarks..."

for test in "${TEST_CASES[@]}"; do
    echo "Running benchmark for: $test"
    
    # Add a separator to the output file for this test run
    echo "" >> "$OUTPUT_FILE"
    echo "--- Test Case: $test ---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Execute the program and APPEND (>>) its output to the results file.
    # The test files are assumed to be in the parent directory.
    ./"$EXEC_NAME" -f "../test_files/$test" >> "$OUTPUT_FILE"
    
    echo "Finished benchmark for: $test"
done

echo "----------------------------------------------------"
echo "All benchmarks complete."


# -- Step 5: Print a final message --
echo "All results have been saved to the file: $OUTPUT_FILE"