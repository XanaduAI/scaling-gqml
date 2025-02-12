import os
import sys
import jax.lax

sys.path.append("..")
from flax_utils import DeepGraphEBM
import yaml
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import iqpopt as iqp
from iqpopt.utils import *
from time import time
from qml_benchmarks.models import DeepEBM
import argparse
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--num-samples', type=int, default=5000)
parser.add_argument('--sampling-time', type=int, help='approx total sampling time (seconds)', default=50000)
parser.add_argument('--chunk-size', type=int, default=1000)
args = parser.parse_args()

dataset_name = args.dataset
model = args.model
sampling_time = args.sampling_time
num_samples = args.num_samples
chunk_size = args.chunk_size

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paper_folder = Path(__file__).resolve().parent.parent
samples_folder = paper_folder / "plots/samples/"

models = [
    "DeepEBM",
    "DeepGraphEBM",
    "RestrictedBoltzmannMachine",
    "IqpSimulator",
    "IqpSimulatorBitflip",
]

model_is_iqp = {
    "IqpSimulator": True,
    "IqpSimulatorBitflip": True,
    "RestrictedBoltzmannMachine": False,
    "DeepEBM": False,
    "DeepGraphEBM": False,
    "True": False,
    "Random": False,
    "RBM": False,
    "GAN": False,
}

############### SETTINGS #################

np.random.seed(666)

num_samples = 5_000
# shoud be fine but maybe puch it to 1000 for small data (only for EBM)

if dataset_name == '2D_ising':
    dataset_path = '../datasets/ising/2d_random_lattice_dataset/ising_4_4_T_3_train.csv'
elif dataset_name == '8_blobs':
    dataset_path = '../datasets/blobs/8_blobs_dataset/16_spins_8_blobs_train.csv'
elif dataset_name == 'spin_glass':
    dataset_path = '../datasets/ising/spin_glass_dataset/ising_spin_glass_N_256_T_0.1_train.csv'
elif dataset_name == 'dwave':
    dataset_path = '../datasets/dwave/dwave_X_train.csv'
elif dataset_name == 'scale_free':
    dataset_path = '../datasets/ising/scale_free_dataset/ising_scale_free_1000_nodes_T_1_train.csv'
elif dataset_name == 'genomic-805':
    dataset_path = '../datasets/genomic/805_SNP_1000G_real_train.csv'
elif dataset_name == 'MNIST':
    dataset_path = '../datasets/MNIST/x_train.csv'

X_test = pd.read_csv(dataset_path, delimiter=',', header=None).to_numpy()
if -1 in X_test:
    X_test = np.array((1 + X_test) // 2)

#########################################

with open(paper_folder / 'training/best_hyperparameters.yaml', 'r') as file:
    hyperparams = yaml.safe_load(file)

params_path = paper_folder / "training/trained_parameters"
dataset_params_files = [f for f in os.listdir(params_path)
                        if os.path.isfile(os.path.join(params_path, f)) and dataset_name in f]

params, params_file = {}, {}
for model_params_file in dataset_params_files:
    for model_name in models:
        if "_"+model_name+"_" in model_params_file:
            with open(params_path / model_params_file, 'rb') as f:
                print(model_params_file)
                params[model_name] = pickle.load(f)
                params_file[model_name] = model_params_file


start = time()
print("Start sampling for", model)

hyper = hyperparams[model][dataset_name]

print(model)
print(hyper)

if model_is_iqp[model]:

    if model == "IqpSimulator":
        bitflip = False
    elif model == "IqpSimulatorBitflip":
        bitflip = True

    if hyper['gates_config']['name'] == 'gates_from_covariance':
        hyper['gates_config']['kwargs']['data'] = X_test[:1000]
    gates = globals()[hyper['gates_config']['name']](
        **hyper['gates_config']['kwargs'])

    iqp_circuit = iqp.IqpSimulator(
        gates=gates, **hyper['model_config'], bitflip=bitflip)
    model_samples = iqp_circuit.sample(params[model], num_samples)

elif model == "DeepEBM":
    deepEBM = DeepEBM(**hyper)
    deepEBM.initialize(X_test[:1000])
    deepEBM.params_ = params[model]

    # avoid storing mcmc chain in memory
    def step_config(i, val):
        return deepEBM.mcmc_step(val, i)[0]

    def sample_config(params, key, x, num_mcmc_steps):
        out = jax.lax.fori_loop(0, num_mcmc_steps, step_config, [params, key, x])
        return out[-1]
    batch_sample = jax.vmap(sample_config, in_axes=(None,0,0, None))


    key = jax.random.PRNGKey(666)
    keys = jax.random.split(key, num_samples+1)
    x_in = jax.random.choice(keys[-1],jnp.array([0,1]), shape=(num_samples,deepEBM.dim))
    t0 = time()
    init_samples = batch_sample(deepEBM.params_, keys[:-1], x_in, 100)
    t1 = time() - t0
    print('time to sample 100 steps: ' + str(t1))
    num_mcmc_steps = int(sampling_time / t1 * 100)
    print('total_mcmc_steps: ' + str(num_mcmc_steps))

    model_samples = batch_sample(deepEBM.params_, keys[:-1], x_in, num_mcmc_steps)
    print(model_samples)
    print(model_samples.shape)

elif model == "DeepGraphEBM":
    G = nx.read_adjlist('../datasets/ising/scale_free_dataset/graph.adjlist')
    deepGraphEBM = DeepGraphEBM(G=G, **hyper)
    deepGraphEBM.initialize(X_test[:1000])
    deepGraphEBM.params_ = params[model]

    t0 = time()
    init_samples = deepGraphEBM.sample(num_samples, num_steps=100, max_chunk_size=chunk_size)
    t1 = time() - t0
    print('time to sample 100 steps: ' + str(t1))
    num_mcmc_steps = int(sampling_time / t1 * 100)
    print('total_mcmc_steps: ' + str(num_mcmc_steps))

    model_samples = deepGraphEBM.sample(num_samples, num_steps=num_mcmc_steps, max_chunk_size=chunk_size)

elif model == "RestrictedBoltzmannMachine":
    rbm = params[model]

    t0 = time()
    init_samples = rbm.sample(num_samples, num_steps=100)
    t1 = time()-t0
    print('time to sample 100 steps: ' +str(t1))
    num_mcmc_steps = int(sampling_time/t1*100)
    print('total_mcmc_steps: ' + str(num_mcmc_steps))

    model_samples = rbm.sample(num_samples, num_steps=num_mcmc_steps)

# with open(samples_folder / f"samples-{params_file[model][7:-4]}.csv", "ab") as f:
#     np.savetxt(f, model_samples, fmt='%d')

np.savetxt(samples_folder /
           f"samples-{params_file[model][7:-4]}.csv", model_samples, fmt='%d')

print("Ended sampling for", model, " - ", time()-start)
