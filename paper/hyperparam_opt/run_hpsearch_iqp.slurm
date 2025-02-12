#!/bin/bash
#SBATCH --job-name=dwave-ebm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_output_%j.txt  # Standard output and error log

# Load the appropriate Python module
module load python/3.11.5
# Activate your virtual environment
source $HOME/env/bin/activate
#python3 -m pip install --no-index --no-cache-dir ../../dist/iqpopt-2024.7.0-py3-none-any.whl
#python3 -m pip install scikit-learn
# python3 -m pip install numpyro
#python3 -m pip install matplotlib
# pip install PennyLane-0.33.1-py3-none-any.whl
#python3 -m pip show iqpopt

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
python run_hyperparameter_search.py --model 'DeepEBM' --dataset 'dwave' --n-jobs 40 --cv 1

# python run_hyperparameter_search.py --model 'DeepGraphEBM' --dataset 'scale_free' --n-jobs 12 --cv 1
