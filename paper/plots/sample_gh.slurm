#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --partition=xanadu-internal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
# SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/job_output_%j.txt  # Standard output and error log

# Load the appropriate Python module
module load python
# Activate your virtual environment
source ~/scratch/jax_cpu/bin/activate

# Run your Python script from the experiments folder
# srun python -u sample_and_save_hpc.py --dataset 'dwave' --model 'RestrictedBoltzmannMachine' --sampling-time 30000 &
#sleep 10
srun python -u sample_and_save_hpc.py --dataset 'dwave' --model 'DeepEBM' --sampling-time 20000 --chunk-size 5000 & 

# Wait for all tasks to finish
wait

echo "Done"
