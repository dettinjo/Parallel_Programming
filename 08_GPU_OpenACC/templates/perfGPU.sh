#!/bin/bash -l
#SBATCH --exclusive
#SBATCH -t 1

# IMPORTANT: execute with exclusive ownership of node

# echo name of host processor (for documentation)
echo "Run on computer:"
hostname
echo

# next variable indicates GPU device number 
export CUDA_VISIBLE_DEVICES=0

echo "Performance Analysis: ncu nvprof"
ncu -c3 -k $1 $2 $3 $4 $5 $6