import numpy as np
import rsmf
import matplotlib.pyplot as plt
import pandas as pd
import plots_config as config

############### SETTINGS #################

aspect_ratio = 0.6

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
        "IqpSimulator",
        "IqpSimulatorBitflip",
        "RestrictedBoltzmannMachine",
        "DeepGraphEBM",
        "True",
        "Random",
    ],
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

for dataset_name in models:
    for model in models[dataset_name]:
        
        formatter = rsmf.setup(config.formatter_setup)
        formatter.figure(aspect_ratio=aspect_ratio)
        cov_matrix = pd.read_csv(config.cov_folder / (dataset_name + "-" + model + "-cov_matrix.csv"), header=None, sep=" ").to_numpy()
        plt.matshow(cov_matrix, cmap='coolwarm')
        plt.clim(vmin=-1, vmax=1)
        plt.title(f"Cov. matrix for {dataset_name} - {config.model_names[model]}")
        plt.colorbar(shrink=0.8)
        plt.savefig(config.figures_folder / (dataset_name + "-cov_matrix-" + model + ".pdf"), dpi=1000, bbox_inches="tight")
        plt.close()
