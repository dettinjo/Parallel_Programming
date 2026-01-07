#!/bin/bash
#SBATCH --job-name=gpu_probe
#SBATCH --output=../results/probe_%j.out
#SBATCH --error=../results/probe_%j.err
#SBATCH --time=00:01:00

# We will use an array job to test different partitions manually 
# or just submit this script multiple times with different --partition arguments.

echo "========================================"
echo "Node: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "========================================"

# 1. Check for Hardware
if command -v nvidia-smi &> /dev/null; then
    echo "SUCCESS: NVIDIA-SMI found!"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "FAILURE: No nvidia-smi found."
fi

# 2. Check for Compiler Module
module purge
if module load nvhpc/21.2 2>/dev/null; then
    echo "SUCCESS: nvhpc/21.2 module loaded."
else
    echo "FAILURE: nvhpc/21.2 module NOT found."
    echo "Available modules:"
    module avail 2>&1 | grep nv
fi
echo "========================================"