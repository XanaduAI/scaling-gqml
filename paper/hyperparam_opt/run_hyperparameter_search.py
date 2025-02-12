# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run hyperparameter search and store results with a command-line script."""

import numpy as np
import sys
import os
import time
import argparse
import logging
logging.getLogger().setLevel(logging.INFO)
from importlib import import_module
import pandas as pd
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from hyperparam_search_utils import read_data, construct_hyperparameter_grid
from hyperparameter_settings import hyper_parameter_settings
from iqp_model_wrapped import IqpSimulator
from iqpopt.gen_qml.utils import median_heuristic, sigma_heuristic
from sklearn.metrics import make_scorer
from sklearn.model_selection import ShuffleSplit
import argparse
import jax

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from flax_utils import DeepGraphEBM

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='The class name of the model')
parser.add_argument('--dataset', type=str, help='The dataset name')
parser.add_argument('--n-jobs', type=int, help='number of parallel jobs')
parser.add_argument('--cv', type=int, help='cross validation folds', default=1)

args = parser.parse_args()

np.random.seed(42)
logging.info('cpu count:' + str(os.cpu_count()))

########## SETTINGS ###########

model_name = args.model
dataset = args.dataset
cross_val = args.cv

##############################

if dataset == '2D_ising':
    dataset_path = '../datasets/ising/2d_random_lattice_dataset/ising_4_4_T_3_train.csv'
elif dataset == '8_blobs':
    dataset_path = '../datasets/blobs/8_blobs_dataset/16_spins_8_blobs_train.csv'
elif dataset == 'spin_glass':
    dataset_path = '../datasets/ising/spin_glass_dataset/ising_spin_glass_N_256_T_3.0_train.csv'
elif dataset == 'dwave':
    dataset_path = '../datasets/dwave/dwave_X_train.csv'
elif dataset == 'genomic-805':
    dataset_path = '../datasets/genomic/805_SNP_1000G_real_train.csv'
elif dataset == 'MNIST':
    dataset_path = '../datasets/MNIST/x_train.csv'
elif dataset == 'scale_free':
    dataset_path = '../datasets/ising/scale_free_dataset/ising_scale_free_1000_nodes_T_1_train.csv'


results_path = '.'

# get the number of score sigmas and delete the entry
n_sigmas_score = hyper_parameter_settings[model_name][dataset]['n_sigmas_score']['val'][0]
if model_name=='IqpSimulator' or model_name=='IqpSimulatorBitflip':
    n_sigmas = hyper_parameter_settings[model_name][dataset]['n_sigmas']['val'][0]

del hyper_parameter_settings[model_name][dataset]['n_sigmas_score']

hyperparam_grid = construct_hyperparameter_grid(hyper_parameter_settings[model_name][dataset])


logging.info(
    "Running hyperparameter search experiment with the following settings\n"
)
logging.info(model_name)
logging.info(dataset_path)
logging.info("Hyperparam grid:"+" ".join([(str(key)+str(":")+str(hyperparam_grid[key])) for key in hyperparam_grid.keys()]))

experiment_path = results_path
results_path = os.path.join(experiment_path, "results")

if not os.path.exists(results_path):
    os.makedirs(results_path)

###################################################################
# Get the classifier, dataset and search methods from the arguments
###################################################################

if model_name == 'IqpSimulator' or model_name == 'IqpSimulatorBitflip':
    Model = IqpSimulator
elif model_name == 'DeepGraphEBM':
    Model = DeepGraphEBM
else:
    Model = getattr(
        import_module("qml_benchmarks.models"),
        args.model
    )
    model_name = Model.__name__

# Run the experiments save the results
train_dataset_filename = os.path.join(dataset_path)
X, y = read_data(train_dataset_filename, labels=False)

if np.any(X == -1):
    X = (X+1)//2

print(X)
print(y)

med = median_heuristic(X[:1000])
print('median heuristic: ' + str(med))


if dataset == '2D_ising' or dataset == '8_blobs':
    score_sigmas = [0.6, 1.3]
    train_sigmas = [0.6, 1.3]
else:
    score_sigmas =  sigma_heuristic(X, n_sigmas_score)
    train_sigmas = None

results_filename_stem = " ".join(
        [model_name + "_" + dataset
         + "_GridSearchCV"])

# # If we have already run this experiment then continue
# if os.path.isfile(os.path.join(results_path, results_filename_stem + ".csv")):
#     msg = "\n================================================================================="
#     msg += "\nResults exist in " + os.path.join(results_path, results_filename_stem + ".csv")
#     msg += "\n================================================================================="
#     logging.warning(msg)
#     sys.exit(msg)

###########################################################################
# Hyperparameter search
###########################################################################

if model_name == 'IqpSimulator':
    estimator = Model(score_sigmas=score_sigmas, train_sigmas=train_sigmas)
elif model_name == 'IqpSimulatorBitflip':
    estimator = Model(bitflip=True, score_sigmas=score_sigmas, train_sigmas=train_sigmas)
elif model_name == 'DeepGraphEBM':
    G = nx.read_adjlist('../datasets/ising/scale_free_dataset/graph.adjlist')
    estimator = Model(G=G)
    estimator.mmd_kwargs['sigma'] = score_sigmas
    estimator.mmd_kwargs['n_steps'] = X.shape[-1]*200
else:
    estimator = Model()
    estimator.mmd_kwargs['sigma'] = score_sigmas
    estimator.mmd_kwargs['n_steps'] = X.shape[-1]*200

if cross_val == 1:
    cross_val = ShuffleSplit(test_size=0.20, n_splits=1, random_state=42)

# To use custom scorers defined by the models
def scorer(estimator, X, y=None):
    return estimator.score(X, y)

gs = GridSearchCV(estimator=estimator,
                  param_grid=hyperparam_grid,
                  scoring=scorer,
                  cv=cross_val,
                  verbose=3,
                  n_jobs=args.n_jobs,
                  refit=False).fit(
    X, y
)

logging.info("Best hyperparams")
logging.info(gs.best_params_)

df = pd.DataFrame.from_dict(gs.cv_results_)
df.to_csv(os.path.join(results_path, results_filename_stem + ".csv"))

best_df = pd.DataFrame(list(gs.best_params_.items()), columns=['hyperparameter', 'best_value'])

# Save best hyperparameters to a CSV file
best_df.to_csv(os.path.join(results_path,
                            results_filename_stem + '-best-hyperparameters.csv'), index=False)
