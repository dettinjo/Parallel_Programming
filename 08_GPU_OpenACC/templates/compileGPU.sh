#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -t 1
#SBATCH --job-name=AHHHH
#SBATCH --output=test_aolin_%j.log
#SBATCH --error=test_aolin_%j.err
#SBATCH --partition=research.q           
#SBATCH --nodelist=aolin-gpu-2                       

# echo name of host processor (for documentation)
echo "Run on computer:"
hostname
echo

# next variable indicates GPU device number 
export CUDA_VISIBLE_DEVICES=0

# install CUDA profiling utilities
module add nvhpc/21.2

echo "Compiling "$1" for GPU compute capability "$2" as file "$3
nvc -fast -acc -gpu=$2 -Minfo=accel $1 -o $3