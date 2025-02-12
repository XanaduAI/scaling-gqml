import os
import yaml
import pickle
import numpy as np
import pandas as pd
import iqpopt as iqp
from iqpopt.utils import *
from time import time
from qml_benchmarks.models import DeepEBM
import plots_config as config

import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from flax_utils import DeepGraphEBM

############### SETTINGS #################

np.random.seed(666)

num_samples = 5_000
num_steps_multiplier = 200
max_chunk_size_ebm = 100 # 100 should be fine but maybe push it to 1000 for small data

models = {
    # '8_blobs': [
        # "IqpSimulator",
        # "IqpSimulatorBitflip",
        # "RestrictedBoltzmannMachine",
        # "DeepEBM",
    # ],
    '2D_ising': [
        # "IqpSimulator",
        "IqpSimulatorBitflip",
        # "RestrictedBoltzmannMachine",
        # "DeepEBM",
    ],
#     'spin_glass': [
#         "IqpSimulatorBitflip",
#         "RestrictedBoltzmannMachine",
#         "DeepEBM",
#     ],
    # 'scale_free': [
    #     "IqpSimulatorBitflip",
    #     "RestrictedBoltzmannMachine",
    #     "DeepGraphEBM",
    # ],
    # 'MNIST': [
    #     "IqpSimulatorBitflip",
    # ],
    # 'genomic-805': [
    #     "IqpSimulatorBitflip",
    # ],
    # 'dwave': [
    #     "IqpSimulatorBitflip",
        # "RestrictedBoltzmannMachine",
        # "DeepEBM",
    # ],
}

#########################################

for dataset_name in models:

    print(dataset_name)
    model_names = models[dataset_name]
    dataset_path = config.test_paths[dataset_name]

    X_test = pd.read_csv(dataset_path, delimiter=',', header=None).to_numpy()
    if -1 in X_test:
        X_test = np.array((1 + X_test) // 2)

    num_steps = num_steps_multiplier * X_test.shape[-1]

    with open(config.hyper_params_path, 'r') as file:
        hyperparams = yaml.safe_load(file)

    dataset_params_files = [f for f in os.listdir(config.params_folder)
                            if os.path.isfile(os.path.join(config.params_folder, f)) and dataset_name in f]

    params, params_file = {}, {}
    for model_params_file in dataset_params_files:
        for model in model_names:
            if "_"+model+"_" in model_params_file:
                with open(config.params_folder / model_params_file, 'rb') as f:
                    print(model_params_file)
                    params[model] = pickle.load(f)
                    params_file[model] = model_params_file

    for model in params.keys():

        start = time()
        print("Start sampling for", model)

        hyper = hyperparams[model][dataset_name]

        if model == "RestrictedBoltzmannMachine":
            rbm = params[model]
            model_samples = rbm.sample(num_samples, num_steps=num_steps)
        
        elif model == "DeepEBM":
            deepEBM = DeepEBM(**hyper)
            deepEBM.initialize(X_test[:1000])
            deepEBM.params_ = params[model]
            model_samples = deepEBM.sample(num_samples, num_steps=num_steps, max_chunk_size=max_chunk_size_ebm)

        elif model == "DeepGraphEBM":
            G = nx.read_adjlist(config.paper_folder / 'datasets/ising/scale_free_dataset/graph.adjlist')
            ebm_model = DeepGraphEBM(G=G, **hyper)
            ebm_model.initialize(X_test[:1000])
            ebm_model.params_ = params_file[model]
            model_samples = ebm_model.sample(num_samples)
            
        else:
            if model == "IqpSimulator":
                bitflip = False
            elif model == "IqpSimulatorBitflip":
                bitflip = True

            os.chdir(config.plots_folder)
            
            if hyper['gates_config']['name'] == 'gates_from_covariance':
                hyper['gates_config']['kwargs']['data'] = X_test[:1000]
            gates = globals()[hyper['gates_config']['name']](**hyper['gates_config']['kwargs'])

            iqp_circuit = iqp.IqpSimulator(
                gates=gates, **hyper['model_config'], bitflip=bitflip)
            model_samples = iqp_circuit.sample(params[model], num_samples)

        np.savetxt(config.samples_folder /f"samples-{params_file[model][7:-4]}.csv", model_samples, fmt='%d')

        print("Ended sampling for", model, " - ", time()-start)
