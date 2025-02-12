import sys
sys.path.append("..")

from flax_utils import *
import networkx as nx
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

np.random.seed(666)

############### SETTINGS #################

dataset_name = 'scale_free'
dataset_path = '../datasets/ising/scale_free_dataset/ising_scale_free_1000_nodes_T_1_train.csv'

G = nx.read_adjlist('../datasets/ising/scale_free_dataset/graph.adjlist')

#########################################

Model = DeepGraphEBM
model_name = Model.__name__

# with open('./best_hyperparameters.yaml', 'r') as file:
with open('./best_hyperparameters.yaml', 'r') as file:
    hyperparams = yaml.safe_load(file)

X_train = pd.read_csv(dataset_path, delimiter=',', header=None).to_numpy()
if -1 in X_train:
    X_train = jnp.array((1 + X_train) // 2, dtype=int)


model = DeepGraphEBM(G=G, **hyperparams[model_name][dataset_name], random_state=np.random.randint(0, 99999))

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
