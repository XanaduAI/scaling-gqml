import pickle
import pandas as pd
import numpy as np
import jax.numpy as jnp
from time import time
from iqpopt.gen_qml.utils import sigma_heuristic
from iqpopt.utils import *
import iqpopt.gen_qml as gen
import iqpopt as iqp
import yaml
import os
import jax
jax.config.update("jax_enable_x64", True)
import plots_config as config

############### SETTINGS #################

np.random.seed(666)
key = jax.random.key(np.random.randint(0, 9999999))

n_ops = 3_000
n_samples = 3_000
max_batch_ops = 3_000
max_batch_samples = 3_000

std_repeats = 20  # number of repeats for standard deviation

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
    'scale_free': [
    #     "IqpSimulator",
        "IqpSimulatorBitflip",
        # "RestrictedBoltzmannMachine",
        # "DeepGraphEBM",
        # "True",
        # "Random",
    ],
    # 'MNIST': [
        # "IqpSimulator",
        # "IqpSimulatorBitflip",
        # "Noise_0.1",
        # "Noise_0.2",
        # "Noise_0.3",
        # "Noise_0.4",
        # "True",
        # "Random",
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
    test_path = config.test_paths[dataset_name]
    train_path = config.train_paths[dataset_name]

    X_train = pd.read_csv(train_path, header=None).to_numpy(dtype=int)
    if -1 in X_train:
        X_train = jnp.array((1 + X_train) // 2, dtype=int)
    print("Train", len(X_train))

    X_test = pd.read_csv(test_path, header=None).to_numpy(dtype=int)
    if -1 in X_test:
        X_test = jnp.array((1 + X_test) // 2, dtype=int)
    print("Test", len(X_test))

    # sigmas = np.array(sigma_heuristic(X_test[:1000], 10)).round(5)
    sigmas = np.linspace(min(config.trained_sigmas[dataset_name]), max(config.trained_sigmas[dataset_name]), 10).round(5)
    print(sigmas)
    print()

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

        if not config.uses_samples[dataset_name].get(model, True):
            hyper = hyperparams[model][dataset_name]

            if model == "IqpSimulator":
                bitflip = False
            elif model == "IqpSimulatorBitflip":
                bitflip = True

            os.chdir(config.plots_folder)
            
            if hyper['gates_config']['name'] == 'gates_from_covariance':
                hyper['gates_config']['kwargs']['data'] = X_test[:1000]
            gates = globals()[hyper['gates_config']['name']](**hyper['gates_config']['kwargs'])

            iqp_circuit = iqp.IqpSimulator(gates=gates, **hyper['model_config'], bitflip=bitflip)
            key, subkey = jax.random.split(key, 2)

            kwargs_models[model]["params"] = jnp.array(params[model], dtype=jnp.float64)
            kwargs_models[model]["iqp_circuit"] = iqp_circuit
            kwargs_models[model]["n_ops"] = n_ops
            kwargs_models[model]["n_samples"] = n_samples
            kwargs_models[model]["key"] = subkey
            kwargs_models[model]["wires"] = hyper['loss_config'].pop('wires', list(range(X_test.shape[-1])))
            kwargs_models[model]["max_batch_ops"] = max_batch_ops
            kwargs_models[model]["max_batch_samples"] = max_batch_samples

        else:
            kwargs_models[model]["model_samples"] = pd.read_csv(
                config.samples_folder / f"samples-{params_file[model][7:-4]}.csv", header=None, delimiter=" ").to_numpy(dtype=int)


    if "GAN" in model_names:
        rng = np.random.default_rng()
        kwargs_models["GAN"] = {
            "model_samples": pd.read_csv(config.samples_folder / "805_SNP_GAN_AG_20000epochs.csv", header=None).to_numpy(),
        }
        rng.shuffle(kwargs_models["GAN"]["model_samples"], axis=0)

    if "RBM" in model_names:
        rng = np.random.default_rng()
        kwargs_models["RBM"] = {
            "model_samples": pd.read_csv(config.samples_folder / "805_SNP_RBM_AG_800Epochs.csv", header=None).to_numpy(),
        }
        rng.shuffle(kwargs_models["RBM"]["model_samples"], axis=0)

    if "True" in model_names:
        kwargs_models["True"] = {
            "model_samples": X_train,
        }

    if "Random" in model_names:
        kwargs_models["Random"] = {
            "model_samples": np.random.randint(0, 2, (len(X_train), X_test.shape[-1])),
        }

    for model_name in model_names:
        if "Noise" in model_name:
            pixel_means =  np.mean(X_test, axis=0)
            error_prob = float(model_name[-3:])
            
            X_noise = X_train.copy()
            for i in range(X_train.shape[0]):
                for j in range(X_train.shape[1]):
                    if np.random.rand() < error_prob:
                        if np.random.rand()<pixel_means[j]:
                            X_noise[i, j] = 1
                        else:
                            X_noise[i, j] = 0
            
            kwargs_models[model_name] = {
                "model_samples": X_noise,
            }
        
    fixed_kwargs_models = kwargs_models.copy()
    keys, model_samples = {}, {}
    for model in fixed_kwargs_models.keys():
        if config.uses_samples[dataset_name].get(model, True):
            model_samples[model] = fixed_kwargs_models[model].pop("model_samples")
        else:
            keys[model] = fixed_kwargs_models[model].pop("key", jax.random.PRNGKey(np.random.randint(0, 99999)))


    for k, model in enumerate(kwargs_models.keys()):
        for j, sigma in enumerate(sigmas):
            losses = []
            if config.uses_samples[dataset_name].get(model, True):
                for i, [ground_truth, model_samples_mmd] in enumerate(zip(np.array_split(X_test, std_repeats), np.array_split(model_samples[model], std_repeats))):
                    current = time()
                    print(f"{dataset_name} dataset. {l+1} / {len(models)}. {model} model. {k+1} / {len(kwargs_models.keys())}. Sigma = {round(sigma,2)}. {j+1} / {len(sigmas)}. Start iteration {i+1} / {std_repeats}. ", end="", flush=True)
                    losses.append(gen.mmd_loss_samples(**fixed_kwargs_models[model], ground_truth=ground_truth, model_samples=model_samples_mmd, sigma=sigma))
                    print(f"Ended in {round(time() - current, 2)} sec.", flush=True)
            else:
                for i, ground_truth in enumerate(np.array_split(X_test, std_repeats)):
                    current = time()
                    print(f"{dataset_name} dataset. {l+1} / {len(models)}. {model} model. {k+1} / {len(kwargs_models.keys())}. Sigma = {round(sigma,2)}. {j+1} / {len(sigmas)}. Start iteration {i+1} / {std_repeats}. ", end="", flush=True)
                    keys[model], subkey = jax.random.split(keys[model], 2)
                    losses.append(gen.mmd_loss_iqp(**fixed_kwargs_models[model], ground_truth=ground_truth, sigma=sigma, key=subkey))
                    print(f"Ended in {round(time() - current, 2)} sec.", flush=True)
            
            print()
            losses = np.array(losses)

            loss = np.mean(losses)
            loss_std = np.std(losses, ddof=1)/np.sqrt(len(losses))
            
            print(loss, loss_std)
            print()

            loss_tbl = {}
            loss_tbl["sigma"] = [sigma]
            loss_tbl["loss"] = [loss]
            loss_tbl["loss_std"] = [loss_std]
            loss_tbl["model"] = [model]

            loss_tbl = pd.DataFrame(loss_tbl)

            old_loss_files = [f for f in os.listdir(config.tables_folder)
                            if os.path.isfile(os.path.join(config.tables_folder, f))
                            and dataset_name in f
                            and 'mmd_loss' in f]

            if len(old_loss_files) >= 1:
                loss_tbl_old = pd.read_csv(config.tables_folder / (dataset_name + "-mmd_loss.csv"))
                loss_tbl = pd.concat([loss_tbl_old, loss_tbl]).drop_duplicates(['model', 'sigma'], keep='last')

            loss_tbl.to_csv(config.tables_folder / (dataset_name + "-mmd_loss.csv"), index=False)
