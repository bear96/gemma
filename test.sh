#!/bin/bash
#SBATCH -J hello-world
#SBATCH --account=thermaltext
#SBATCH --partition=normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --time=0-00:10:00

echo "hello world from..."
hostname
