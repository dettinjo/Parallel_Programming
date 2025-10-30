#!/bin/bash
#SBATCH --job-name=cpuinfo
#SBATCH --output=cpu-info-out
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cuda-ext.q

lscpu