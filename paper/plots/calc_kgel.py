import os
import yaml
import iqpopt as iqp
import iqpopt.gen_qml as gen
from iqpopt.utils import *
from iqpopt.gen_qml.utils import sigma_heuristic
from time import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pickle
import plots_config as config

############### SETTINGS #################

np.random.seed(666)
key = jax.random.key(np.random.randint(0, 9999999))

n_witnesses = 10
n_test_samples = 2_000

n_ops = 6_000
n_samples = 6_000
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
    # 'MNIST': [
    #     "IqpSimulator",
    #     "IqpSimulatorBitflip",
    #     "Noise_0.1",
    #     "Noise_0.2",
    #     "Noise_0.3",
    #     "Noise_0.4",
    #     "True",
    #     "Random",
    # ],
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
    print(dataset_name)
    
    model_names = models[dataset_name]
    train_path = config.train_paths[dataset_name]
    test_path = config.test_paths[dataset_name]
    
    X_train = pd.read_csv(train_path, header=None).to_numpy(dtype=int)
    if -1 in X_train:
        X_train = jnp.array((1 + X_train) // 2, dtype=int)
    print("Train", len(X_train))

    X_test = pd.read_csv(test_path, header=None).to_numpy(dtype=int)
    if -1 in X_test:
        X_test = jnp.array((1 + X_test) // 2, dtype=int)
    print("Test", len(X_test))

    # sigmas = np.array([np.sqrt(median_heuristic(X_test[:1000]))]).round(5)
    # sigmas = np.array(sigma_heuristic(X_test[:1000], 3)).round(5)
    sigmas = np.linspace(min(config.trained_sigmas[dataset_name]), max(config.trained_sigmas[dataset_name]), 3).round(5)
    

    with open(config.hyper_params_path, 'r') as file:
        hyperparams = yaml.safe_load(file)

    dataset_params_files = [f for f in os.listdir(config.params_folder)
                            if os.path.isfile(os.path.join(config.params_folder, f)) and dataset_name in f]

    params = {}
    params_file = {}
    for model_params_file in dataset_params_files:
        for model in model_names:
            if "_"+model+"_" in model_params_file:
                with open(config.params_folder / model_params_file, 'rb') as f:
                    params[model] = pickle.load(f)
                    params_file[model] = model_params_file


    kwargs_models = {}
    for model in params.keys():
        kwargs_models[model] = {}
        kwargs_models[model]["ground_truth"] = X_test[:n_test_samples]
        kwargs_models[model]["witnesses"] = X_test[-n_witnesses:]

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
                hyper['gates_config']['kwargs']['data'] = X_test[:n_test_samples]
            gates = globals()[hyper['gates_config']['name']](**hyper['gates_config']['kwargs'])

            iqp_circuit = iqp.IqpSimulator(
                gates=gates, **hyper['model_config'], bitflip=bitflip)
            key, subkey = jax.random.split(key, 2)

            kwargs_models[model]["params"] = params[model]
            kwargs_models[model]["iqp_circuit"] = iqp_circuit
            kwargs_models[model]["n_ops"] = n_ops
            kwargs_models[model]["n_samples"] = n_samples
            kwargs_models[model]["key"] = subkey
            kwargs_models[model]["wires"] = list(range(X_test.shape[-1]))
            kwargs_models[model]["max_batch_ops"] = max_batch_ops
            kwargs_models[model]["max_batch_samples"] = max_batch_samples


    if "GAN" in model_names:
        kwargs_models["GAN"] = {
            "ground_truth": X_test[:n_test_samples],
            "model_samples": pd.read_csv(config.samples_folder / "805_SNP_GAN_AG_20000epochs.csv", header=None).to_numpy(),
            "witnesses": X_test[-n_witnesses:],
        }

    if "RBM" in model_names:
        kwargs_models["RBM"] = {
            "ground_truth": X_test[:n_test_samples],
            "model_samples": pd.read_csv(config.samples_folder / "805_SNP_RBM_AG_800Epochs.csv", header=None).to_numpy(),
            "witnesses": X_test[-n_witnesses:],
        }

    if "True" in model_names:
        kwargs_models["True"] = {
            "ground_truth": X_test[:n_test_samples],
            "model_samples": X_train[:n_test_samples],
            "witnesses": X_test[-n_witnesses:],
        }

    if "Random" in model_names:
        kwargs_models["Random"] = {
            "ground_truth": X_test[:n_test_samples],
            "model_samples": np.random.randint(0, 2, (n_test_samples, X_test.shape[-1])),
            "witnesses": X_test[-n_witnesses:],
        }

    for model_name in model_names:
        if "Noise" in model_name:
            error_prob = float(model_name[-3:]) * np.mean(X_test, axis=0)
            
            X_noise = X_train.copy()
            for i in range(X_train.shape[0]):
                for j in range(X_train.shape[1]):
                    if np.random.rand() < error_prob[j]:
                        if X_noise[i,j] == 0:
                            X_noise[i,j] = 1
                        else:
                            X_noise[i,j] = 0
            
            kwargs_models[model_name] = {
                "ground_truth": X_test[:n_test_samples],
                "model_samples": X_noise,
                "witnesses": X_test[-n_witnesses:],
            }
    
    fixed_kwargs_models = kwargs_models.copy()
    keys, model_samples, labels_truth = {}, {}, {}
    for model in fixed_kwargs_models.keys():
        if not config.uses_samples[dataset_name].get(model, True):
            keys[model] = fixed_kwargs_models[model].pop("key", jax.random.PRNGKey(np.random.randint(0, 99999)))

    for k, model in enumerate(kwargs_models.keys()):
        for j, sigma in enumerate(sigmas):

            try:
                kgel_tbl = {}
                current = time()
                print(f"{dataset_name} dataset. {l+1} / {len(models)}. {model} model. {k+1} / {len(kwargs_models.keys())}. Sigma = {round(sigma,2)}. {j+1} / {len(sigmas)}. ", end="", flush=True)
                
                if config.uses_samples[dataset_name].get(model, True):
                    kgel, pi = gen.kgel_opt_samples(**fixed_kwargs_models[model], sigma=sigma)
                else:
                    keys[model], subkey = jax.random.split(keys[model], 2)
                    kgel, pi = gen.kgel_opt_iqp(**fixed_kwargs_models[model], sigma=sigma, key=subkey)
                
                print(f"Ended in {round(time() - current, 2)} sec.", flush=True)
                
                kgel_tbl["sigma"] = [sigma]
                kgel_tbl["kgel"] = [kgel]
                kgel_tbl["pi"] = [pi]
                kgel_tbl["model"] = [model]

                kgel_tbl = pd.DataFrame(kgel_tbl)
                
                old_kgel_files = [f for f in os.listdir(config.tables_folder)
                                if os.path.isfile(os.path.join(config.tables_folder, f))
                                and dataset_name in f
                                and 'kgel' in f
                                and 'pkl' in f]

                if len(old_kgel_files) >= 1:
                    with open(config.tables_folder / (dataset_name + "-kgel.pkl"), 'rb') as f:
                        kgel_tbl_old = pickle.load(f)
                    
                    kgel_tbl = pd.concat([kgel_tbl_old, kgel_tbl]).drop_duplicates(['model', 'sigma'], keep='last')

                with open(config.tables_folder / (dataset_name + "-kgel.pkl"), 'wb') as f:
                    pickle.dump(kgel_tbl, f)
                    
            except Exception as e:
                print(e)

        print()
