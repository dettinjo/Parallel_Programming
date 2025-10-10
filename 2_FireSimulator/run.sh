#!/bin/bash
#SBATCH -N 1
##SBATCH -n 12
#SBATCH --exclusive
#SBATCH --distribution=cyclic
##SBATCH --partition=nodo.q # Wilma node
#SBATCH --partition=cuda-ext.q # Aolin node
#SBATCH --output=out_sbatch.txt
#SBATCH --error=err_sbatch.txt

# To run this script in a node, write in terminal: sbatch run.sh

module add gcc

gcc -Ofast -o exec extinguishing.c -lm
# Example run
./exec -f test_files/test1

