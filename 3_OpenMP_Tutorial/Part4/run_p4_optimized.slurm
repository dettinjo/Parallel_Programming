#!/bin/bash
#SBATCH --job-name=P4_Optimized
#SBATCH --output=p4_slurm_log_%j.out
#SBATCH --error=p4_slurm_log_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:02:00
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=cuda-ext.q
#SBATCH --nodelist=aolin-gpu-1

SOURCE_FILE="PrgPAR_P4.c"
EXEC_NAME="PrgPAR_P4"
OUTPUT_FILE="p4_execution_output.txt"
NUM_THREADS=2

echo "Running Part 4 (P4) Optimized Parallel on node: $(hostname)"
echo "----------------------------------------------------"

echo "Compiling $SOURCE_FILE with -fopenmp..."
gcc -Ofast -fno-inline -fopenmp "$SOURCE_FILE" -o "$EXEC_NAME" -lm
if [ $? -ne 0 ]; then echo "Compilation failed!"; exit 1; fi
echo "Compilation successful."
echo "----------------------------------------------------"

export OMP_NUM_THREADS=$NUM_THREADS
echo "OMP_NUM_THREADS set to $NUM_THREADS"
echo "Running 'time ./${EXEC_NAME}' and saving output to $OUTPUT_FILE..."

{ time ./"$EXEC_NAME" ; } 2>&1 > "$OUTPUT_FILE"

echo "----------------------------------------------------"
echo "Job finished. The output is in '$OUTPUT_FILE'."