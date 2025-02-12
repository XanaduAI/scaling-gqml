#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_output_%j.txt  # Standard output and error log

# Load the appropriate Python module
module load python/3.11.5
# Activate your virtual environment
source $HOME//env/bin/activate

# Run your Python script from the experiments folder
srun  --exclusive -n 1 -c 20 --mem-per-cpu 4048 python -u sample_and_save_hpc.py --dataset 'scale_free' --model 'RestrictedBoltzmannMachine' --sampling-time 50000 &
sleep 10
srun  --exclusive -n 1 -c 20 --mem-per-cpu 4048 python -u sample_and_save_hpc.py --dataset 'scale_free' --model 'DeepGraphEBM' --sampling-time 50000 --chunk-size 500 & 

# Wait for all tasks to finish
wait

echo "Done"
