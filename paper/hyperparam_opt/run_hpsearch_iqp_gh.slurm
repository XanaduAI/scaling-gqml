#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --partition=xanadu-internal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --output=logs/job_output_%j.txt  # Standard output and error log

module load python
# Activate your virtual environment
source ~/scratch/env/bin/activate

#echo $SLURM_TMPDIR
#which python3
#which pip
#which pip3

# Run your Python script from the experiments folder
# python run_hyperparameter_search.py --model 'IqpSimulatorBitflip' --dataset '2D_ising' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulator' --dataset '2D_ising' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulatorBitflip' --dataset '8_blobs' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulator' --dataset '8_blobs' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulator' --dataset 'spin_glass' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulatorBitflip' --dataset 'spin_glass' --n-jobs 30 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulator' --dataset 'dwave' --n-jobs 8 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulatorBitflip' --dataset 'dwave' --n-jobs 12 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulator' --dataset 'genomic-805' --n-jobs 5 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulatorBitflip' --dataset 'genomic-805' --n-jobs 4 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulator' --dataset 'MNIST' --n-jobs 8 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulatorBitflip' --dataset 'MNIST' --n-jobs 5 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulator' --dataset 'scale_free' --n-jobs 20 --cv 1
# python run_hyperparameter_search.py --model 'IqpSimulatorBitflip' --dataset 'scale_free' --n-jobs 20 --cv 1

# python run_hyperparameter_search.py --model 'RestrictedBoltzmannMachine' --dataset '2D_ising' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'RestrictedBoltzmannMachine' --dataset '8_blobs' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'RestrictedBoltzmannMachine' --dataset 'spin_glass' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'RestrictedBoltzmannMachine' --dataset 'dwave' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'RestrictedBoltzmannMachine' --dataset 'MNIST' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'RestrictedBoltzmannMachine' --dataset 'scale_free' --n-jobs 40 --cv 1

# python run_hyperparameter_search.py --model 'DeepEBM' --dataset '2D_ising' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'DeepEBM' --dataset '8_blobs' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'DeepEBM' --dataset 'spin_glass' --n-jobs 40 --cv 1
# python run_hyperparameter_search.py --model 'DeepEBM' --dataset 'dwave' --n-jobs 40 --cv 1

# python run_hyperparameter_search.py --model 'DeepGraphEBM' --dataset 'scale_free' --n-jobs 12 --cv 1
