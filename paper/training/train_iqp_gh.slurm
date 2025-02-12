#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --partition=xanadu-internal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job_output_%j.txt  # Standard output and error log

# Load the appropriate Python module
module load python
module load cuda_default
# Activate your virtual environment
source ~/scratch/env/bin/activate

nvidia-smi
pip list
pip install -U "jax[cuda12]"

# Run your Python script from the experiments folder
srun python -u train_iqp_hpc.py --dataset 'genomic-805' --bitflip False &
sleep 10

# Wait for all tasks to finish
wait

echo "Done"
