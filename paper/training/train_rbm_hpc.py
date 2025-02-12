import yaml
from qml_benchmarks.models.energy_based_model import RestrictedBoltzmannMachine
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import iqpopt.training as train
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import argparse
import pickle

np.random.seed(666)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset = args.dataset
dataset_name = args.dataset

############### SETTINGS #################

if dataset == '8_blobs':
    dataset_path = '../datasets/blobs/8_blobs_dataset/16_spins_8_blobs_train.csv'
elif dataset == '2D_ising':
    dataset_path = '../datasets/ising/2d_random_lattice_dataset/ising_4_4_T_3_train.csv'
elif dataset == 'spin_glass':
    dataset_path = '../datasets/ising/spin_glass_dataset/ising_spin_glass_N_256_T_0.1_train.csv'
elif dataset_name == 'dwave':
    dataset_path = '../datasets/dwave/dwave_X_train.csv'
elif dataset == 'scale_free':
    dataset_path = '../datasets/ising/scale_free_dataset/ising_scale_free_1000_nodes_T_1_train.csv'
elif dataset == 'MNIST':
    dataset_path = '../datasets/MNIST/x_train.csv'

#########################################

Model = RestrictedBoltzmannMachine
model_name = Model.__name__

# with open('./best_hyperparameters.yaml', 'r') as file:
with open('./best_hyperparameters.yaml', 'r') as file:
    hyperparams = yaml.safe_load(file)

X_train = pd.read_csv(dataset_path, delimiter=',', header=None).to_numpy()
if -1 in X_train:
    X_train = jnp.array((1 + X_train) // 2, dtype=int)

print(X_train)

model = Model(**hyperparams[model_name][dataset_name], random_state=np.random.randint(0, 99999))

# train the model
model.fit(X_train)

# (sklearn RBM does not give the loss history...)
# save loss plots
# sns.lineplot(model.loss_history_, label='train')
# plt.xlabel('training step')
# plt.ylabel('contrastive divergence value')
# plt.savefig(f'./loss_plots/loss_plot_{model_name}_{dataset_name}.png', dpi=300)
# plt.clf()
#
# sns.lineplot(model.loss_history_, label='train')
# plt.xlabel('training step')
# plt.ylabel('contrastive divergence value')
# plt.yscale('log')
# plt.savefig(f'./loss_plots/loss_plot_{model_name}_{dataset_name}_log.png', dpi=300)


# save the losses
# np.savetxt(f'./loss_plots/train_losses_{model_name}_{dataset_name}.csv', model.loss_history_, delimiter=',')

# save the final model
with open(f'./trained_parameters/params_{model_name}_{dataset_name}.pkl', 'wb') as f:
    pickle.dump(model, f)
