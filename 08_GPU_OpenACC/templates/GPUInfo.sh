#!/bin/bash -l

# no need to claim computer in exclusive mode

echo "GPU info for following computer node"
hostname
echo

# install NVIDIA tools
module add nvhpc/21.2

# list GPU (accelerator) info
export CUDA_VISIBLE_DEVICES=0,1
nvaccelinfo
echo

# list GPU device info (CUDA utility)
nvidia-smi

echo
echo
echo "CPU info for the computer node"
echo
nvc -V
echo
lscpu
echo
lspci