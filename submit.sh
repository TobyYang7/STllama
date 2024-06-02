#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=dcu:4
#SBATCH -p kshdnormal02

# conda install openssl=1.1.1 -c conda-forge

source ~/env.sh

bash train.sh
