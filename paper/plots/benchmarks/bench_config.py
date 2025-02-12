import numpy as np
from pathlib import Path

evaluating = "Precision" # "Time" or "Precision"
machine = "cpu" # "cpu" or "gpu"

# runtime plots
# funcs = ["op_expval"] # either ["MMD loss", "Loss gradient"] or ["op_expval"]
# variable = "n_qubits"
# s_lines = "n_gates"
# v_lines = [1_000, 100_000, 1_000_000]

# precision plots
# runtime
funcs = ["op_expval"] # either ["MMD loss", "Loss gradient"] or ["op_expval"]
variable = "n_samples"
s_lines = "n_qubits"
v_lines = [100, 1_000, 10_000]

#variable = "n_qubits"
#s_lines = "n_gates"
#v_lines = [1_000, 10_000]]

# Define the x axis of the plot
min = 100
max = 100_000
# max = 1_000_000
n_points = 20

# Seed for reproducibility if desired
np.random.seed(None)

# Define the default parameters of all plots
default = {
    "n_qubits": 100,
    "n_gates": 100_000,
    "sparse": False,
    "n_samples": 10000,
    "n_ops": 1000,
    "max_batch_samples": 10_000,
    "max_batch_ops": 10_000,
}

# Parameters for MMD loss
n_test = 1000
sigma = 0.1

# Plot settings
formatter_setup = r"\documentclass[a4paper,twocolumn,notitlepage,nofootinbib]{revtex4-2}"
aspect_ratio = 0.6

# Files save folder
bench_folder = Path(__file__).resolve().parent
filename = f"{funcs[0]} - {evaluating} vs {variable} - {machine}"
