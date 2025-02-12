#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_output_%j.txt  # Standard output and error log

# Load the appropriate Python module
module load python/3.11.5
# Activate your virtual environment
source $HOME/env/bin/activate

# Run your Python script from the experiments folder
srun  --exclusive -n 1 -c 20 --mem-per-cpu 2048 python train_iqp_hpc.py --dataset 'spin_glass' --bitflip False --turbo 50 &
sleep 30
srun  --exclusive -n 1 -c 20 --mem-per-cpu 2048 python train_iqp_hpc.py --dataset 'spin_glass' --bitflip True --turbo 50 &
sleep 30
srun  --exclusive -n 1 -c 20 --mem-per-cpu 2048 python train_iqp_hpc.py --dataset 'genomic-805' --bitflip True &
sleep 30
srun  --exclusive -n 1 -c 20 --mem-per-cpu 2048 python train_iqp_hpc.py --dataset 'genomic-805' --bitflip False &

# Wait for all tasks to finish
wait

echo "Done"
