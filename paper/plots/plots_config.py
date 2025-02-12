from pathlib import Path
import yaml

# Formatter settup from rsmf
formatter_setup = r"\documentclass[a4paper,twocolumn,notitlepage,nofootinbib]{revtex4-2}"

# Directory paths
paper_folder = Path(__file__).resolve().parent.parent
plots_folder = paper_folder / "plots/"

hyper_params_path = paper_folder / 'training/best_hyperparameters.yaml'
params_folder = paper_folder / "training/trained_parameters"


samples_folder = plots_folder / "samples/"
tables_folder = plots_folder / "tables/"
figures_folder = plots_folder / "figures/"
j_folder = plots_folder / "j_matrix/"
cov_folder = plots_folder / "cov_matrix/"

test_paths = {
    '8_blobs': paper_folder / 'datasets/blobs/8_blobs_dataset/16_spins_8_blobs_test.csv',
    '2D_ising': paper_folder / 'datasets/ising/2d_random_lattice_dataset/ising_4_4_T_3_test.csv',
    'spin_glass': paper_folder / 'datasets/ising/spin_glass_dataset/ising_spin_glass_N_256_T_0.1_test.csv',
    'scale_free': paper_folder / 'datasets/ising/scale_free_dataset/ising_scale_free_1000_nodes_T_1_test.csv',
    'MNIST': paper_folder / 'datasets/MNIST/x_test.csv',
    'genomic-805': paper_folder / 'datasets/genomic/805_SNP_1000G_real_test.csv',
    'dwave': paper_folder / 'datasets/dwave/dwave_X_test.csv',
}

train_paths = {
    '8_blobs': paper_folder / 'datasets/blobs/8_blobs_dataset/16_spins_8_blobs_train.csv',
    '2D_ising': paper_folder / 'datasets/ising/2d_random_lattice_dataset/ising_4_4_T_3_train.csv',
    'spin_glass': paper_folder / 'datasets/ising/spin_glass_dataset/ising_spin_glass_N_256_T_0.1_train.csv',
    'scale_free': paper_folder / 'datasets/ising/scale_free_dataset/ising_scale_free_1000_nodes_T_1_train.csv',
    'MNIST': paper_folder / 'datasets/MNIST/x_train.csv',
    'genomic-805': paper_folder / 'datasets/genomic/805_SNP_1000G_real_train.csv',
    'dwave': paper_folder / 'datasets/dwave/dwave_X_train.csv',
}

test_label_paths = {
    "8_blobs": paper_folder / 'datasets/blobs/8_blobs_dataset/16_spins_8_blobs_labels_test.csv',
    "MNIST": paper_folder / 'datasets/MNIST/y_test.csv',
}

# Dictionary that indicates if the model uses samples, being True by default (if the model is not here).
uses_samples = {
    '8_blobs': {
        "IqpSimulator": False,
        "IqpSimulatorBitflip": True,
    },
    '2D_ising': {
        "IqpSimulator": False,
        "IqpSimulatorBitflip": True,
    },
    'spin_glass': {
        "IqpSimulator": False,
        "IqpSimulatorBitflip": True,
    },
    'scale_free': {
        "IqpSimulator": False,
        "IqpSimulatorBitflip": True,
    },
    'MNIST': {
        "IqpSimulator": False,
        "IqpSimulatorBitflip": True,
    },
    'genomic-805': {
        "IqpSimulator": False,
        "IqpSimulatorBitflip": True,
    },
    'dwave': {
        "IqpSimulator": False,
        "IqpSimulatorBitflip": True,
    },
}

# Dictionary with the colors that will be used in the plots to identify each model
model_colors = {
    "IqpSimulator": "#027ab0",
    "IqpSimulatorBitflip": "#1ebecd",
    "RestrictedBoltzmannMachine": "#49997c",
    "DeepEBM": "#ae3918",
    "DeepGraphEBM": "#ae3918",
    "True": "black",
    "Random": "#d19c2f",
    "RBM": "#49997c",
    "GAN": "#ae3918",
    "Noise_0.1": "0.2",
    "Noise_0.2": "0.4",
    "Noise_0.3": "0.6",
    "Noise_0.4": "0.8",
}

# Dictionary with the names that will be used in the plots to identify each model
model_names = {
    "IqpSimulator": "IQP",
    "IqpSimulatorBitflip": "Bitflip",
    "RestrictedBoltzmannMachine": "RBM",
    "DeepEBM": "DeepEBM",
    "DeepGraphEBM": "DeepGraphEBM",
    "True": "True",
    "Random": "Random",
    "RBM": "RBM",
    "GAN": "GAN",
    "Noise_0.1": "Noise 0.1",
    "Noise_0.2": "Noise 0.2",
    "Noise_0.3": "Noise 0.3",
    "Noise_0.4": "Noise 0.4",
}

with open(hyper_params_path, 'r') as file:
    hyperparams = yaml.safe_load(file)

trained_sigmas = {
    '8_blobs': hyperparams["IqpSimulator"]['8_blobs']["loss_config"]["sigma"],
    '2D_ising': hyperparams["IqpSimulator"]['2D_ising']["loss_config"]["sigma"],
    'spin_glass': hyperparams["IqpSimulator"]['spin_glass']["loss_config"]["sigma"],
    'scale_free': hyperparams["IqpSimulator"]['scale_free']["loss_config"]["sigma"],
    'MNIST': hyperparams["IqpSimulator"]['MNIST']["loss_config"]["sigma"],
    'genomic-805': hyperparams["IqpSimulator"]['genomic-805']["loss_config"]["sigma"],
    'dwave': hyperparams["IqpSimulator"]['dwave']["loss_config"]["sigma"],
}