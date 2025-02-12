import sys
sys.path.append("..")

from flax_utils import *
import yaml
from qml_benchmarks.models.energy_based_model import DeepEBM
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import iqpopt.training as train
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle

plt.rcParams['agg.path.chunksize'] = 10000

np.random.seed(666)

############### SETTINGS #################

# dataset_name = '8_blobs'
# dataset_path = '../datasets/blobs/8_blobs_dataset/16_spins_8_blobs_train.csv'

dataset_name = '2D_ising'
dataset_path = '../datasets/ising/2d_random_lattice_dataset/ising_4_4_T_3_train.csv'

# dataset_name = 'spin_glass'
# dataset_path = '../datasets/ising/spin_glass_dataset/ising_spin_glass_N_256_T_0.1_train.csv'

# dataset_name = 'dwave'
# dataset_path = '../datasets/dwave/dwave_X_train.csv'


#########################################

Model = DeepEBM
model_name = Model.__name__

# with open('./best_hyperparameters.yaml', 'r') as file:
with open('./best_hyperparameters.yaml', 'r') as file:
    hyperparams = yaml.safe_load(file)

X_train = pd.read_csv(dataset_path, delimiter=',', header=None).to_numpy()
if -1 in X_train:
    X_train = jnp.array((1 + X_train) // 2, dtype=int)


model = DeepEBM(**hyperparams[model_name][dataset_name], random_state=np.random.randint(0, 99999))

# train the model
model.fit(X_train)

# save loss plots
sns.lineplot(model.loss_history_, label='train')
plt.xlabel('training step')
plt.ylabel('contrastive divergence value')
plt.savefig(f'./loss_plots/loss_plot_{model_name}_{dataset_name}.png', dpi=300)

# save loss plots
sns.lineplot(model.loss_history_, label='train')
plt.xlabel('training step')
plt.ylabel('contrastive divergence value')
plt.yscale('log')
plt.savefig(f'./loss_plots/loss_plot_{model_name}_{dataset_name}_log.png', dpi=300)

# save the losses
np.savetxt(f'./loss_plots/train_losses_{model_name}_{dataset_name}.csv', model.loss_history_, delimiter=',')

# save the final parameters
with open(f'./trained_parameters/params_{model_name}_{dataset_name}.pkl', 'wb') as f:
    pickle.dump(model.params_, f)
