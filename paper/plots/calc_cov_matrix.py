import os
import yaml
import iqpopt as iqp
from iqpopt.utils import *
from iqpopt.utils import construct_convariance_matrix
from time import time
import jax
import numpy as np
import pandas as pd
import pickle
import plots_config as config

############### SETTINGS #################

np.random.seed(666)
key = jax.random.key(np.random.randint(0, 9999999))

n_samples = 3_000
max_batch_ops = 3_000
max_batch_samples = 3_000

models = {
    # '8_blobs': [
    #     "IqpSimulator",
    #     "IqpSimulatorBitflip",
    #     "RestrictedBoltzmannMachine",
    #     "DeepEBM",
    #     "True",
    #     "Random",
    # ],
    # '2D_ising': [
    #     "IqpSimulator",
    #     "IqpSimulatorBitflip",
    #     "RestrictedBoltzmannMachine",
    #     "DeepEBM",
    #     "True",
    #     "Random",
    # ],
    # 'spin_glass': [
    #     "IqpSimulator",
    #     "IqpSimulatorBitflip",
    #     "RestrictedBoltzmannMachine",
    #     "DeepEBM",
    #     "True",
    #     "Random",
    # ],
    # 'scale_free': [
    #     "IqpSimulator",
    #     "IqpSimulatorBitflip",
    #     "RestrictedBoltzmannMachine",
    #     "DeepGraphEBM",
    #     "True",
    #     "Random",
    # ],
    'MNIST': [
    #     "IqpSimulator",
    #     "IqpSimulatorBitflip",
        "Noise_0.1",
        "Noise_0.2",
        "Noise_0.3",
        "Noise_0.4",
    #     "True",
    #     "Random",
    ],
    # 'genomic-805': [
    #     "IqpSimulator",
    #     "IqpSimulatorBitflip",
    #     "True",
    #     "Random",
    #     "RBM",
    #     "GAN",
    # ],
    # 'dwave': [
        # "IqpSimulator",
        # "IqpSimulatorBitflip",
        # "RestrictedBoltzmannMachine",
        # "DeepEBM",
        # "True",
        # "Random",
    # ],
}

#########################################

for l, dataset_name in enumerate(models):
    
    model_names = models[dataset_name]
    test_path = config.test_paths[dataset_name]

    X_test = pd.read_csv(test_path, header=None).to_numpy(dtype=int)

    with open(config.hyper_params_path, 'r') as file:
        hyperparams = yaml.safe_load(file)

    dataset_params_files = [f for f in os.listdir(config.params_folder)
                            if os.path.isfile(os.path.join(config.params_folder, f))
                            and dataset_name in f]

    params = {}
    params_file = {}
    for model in model_names:
        print(model)
        for model_params_file in dataset_params_files:
            if "_"+model+"_" in model_params_file:
                with open(config.params_folder / model_params_file, 'rb') as f:
                    print(model_params_file)
                    params[model] = pickle.load(f)
                    params_file[model] = model_params_file
        print()

    kwargs_models = {}
    for model in params.keys():
        kwargs_models[model] = {}
        
        if config.uses_samples[dataset_name].get(model, True):
            kwargs_models[model]["model_samples"] = pd.read_csv(config.samples_folder / f"samples-{params_file[model][7:-4]}.csv", header=None, delimiter=" ").to_numpy(dtype=int)
        else:
            hyper = hyperparams[model][dataset_name]

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
            key, subkey = jax.random.split(key, 2)

            kwargs_models[model]["circuit"] = iqp_circuit
            kwargs_models[model]["params"] = params[model]
            kwargs_models[model]["n_samples"] = n_samples
            kwargs_models[model]["key"] = subkey
            kwargs_models[model]["max_batch_ops"] = max_batch_ops
            kwargs_models[model]["max_batch_samples"] = max_batch_samples


    if "GAN" in model_names:
        kwargs_models["GAN"] = {
            "model_samples": pd.read_csv(config.samples_folder / "805_SNP_GAN_AG_20000epochs.csv", header=None).to_numpy(),
        }

    if "RBM" in model_names:
        kwargs_models["RBM"] = {
            "model_samples": pd.read_csv(config.samples_folder / "805_SNP_RBM_AG_800Epochs.csv", header=None).to_numpy(),
        }

    if "True" in model_names:
        kwargs_models["True"] = {
            "model_samples": X_test,
        }

    if "Random" in model_names:
        kwargs_models["Random"] = {
            "model_samples": np.random.randint(0, 2, (len(X_test), X_test.shape[-1])),
        }

    for model_name in model_names:
        if "Noise" in model_name:
            error_prob = float(model_name[-3:]) * np.mean(X_test, axis=0)
            
            X_noise = X_test.copy()
            for i in range(X_test.shape[0]):
                for j in range(X_test.shape[1]):
                    if np.random.rand() < error_prob[j]:
                        if X_noise[i,j] == 0:
                            X_noise[i,j] = 1
                        else:
                            X_noise[i,j] = 0
            
            kwargs_models[model_name] = {
                "model_samples": X_noise,
            }

    for k, model in enumerate(kwargs_models.keys()):
        current = time()
        print(f"{dataset_name} dataset. {l+1} / {len(models)}. {model} model. {k+1} / {len(kwargs_models.keys())}. Start covariance matrix calculation. ", end="", flush=True)

        if config.uses_samples[dataset_name].get(model, True):
            samples = kwargs_models[model]["model_samples"]
            if 0 in samples:
                samples = 1 - 2*samples

            matrix = np.cov(samples.T)
            
        else:
            matrix = construct_convariance_matrix(**kwargs_models[model])

        print(f"Ended in {round(time() - current, 2)} sec.", flush=True)
        print()

        np.savetxt(config.cov_folder / (dataset_name + "-" + model + "-cov_matrix.csv"), matrix)
