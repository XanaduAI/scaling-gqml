import rsmf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from iqpopt.gen_qml.utils import sigma_heuristic
import plots_config as config


############### SETTINGS #################

squishing_factor = 60
aspect_ratio = 0.6

datasets = [
    # "8_blobs",
    # "2D_ising",
    # "spin_glass",
    "scale_free",
    # "MNIST",
    # "genomic-805",
    # "dwave",
]

#########################################

for dataset_name in datasets:

    dataset_path = config.test_paths[dataset_name]
    
    X_test = pd.read_csv(dataset_path, delimiter=',', header=None).to_numpy(dtype=int)
    if -1 in X_test:
        X_test = np.array((1 + X_test) // 2, dtype=int)

    # plt_sigmas = np.array(sigma_heuristic(X_test[:1000], 10)).round(3)
    plt_sigmas = np.linspace(min(config.trained_sigmas[dataset_name]), max(config.trained_sigmas[dataset_name]), 10).round(3)

    loss_tbl = pd.read_csv(config.tables_folder / (dataset_name + "-mmd_loss.csv"))
    loss_tbl = loss_tbl.round({'sigma': 3})
    
    plt_sigmas = pd.Series(plt_sigmas, name='sigma')
    loss_tbl = pd.merge(loss_tbl, plt_sigmas, "inner")

    models = loss_tbl["model"].unique()
    sigmas = loss_tbl["sigma"].unique()

    formatter = rsmf.setup(config.formatter_setup)
    formatter.figure(aspect_ratio=aspect_ratio)

    # squishing_factor: the higher, the more squished together the lines will be with their sigma
    for i, model in enumerate(models):
        df = loss_tbl[loss_tbl["model"] == model]
        width = (sigmas[-1]-sigmas[0])/(len(sigmas)-1)
        x = df["sigma"] - width/2 + (i+1+squishing_factor)*width/(len(models)+1+squishing_factor*2)
        if "Noise" in model:
            plt.plot(x, df["loss"], c=config.model_colors[model], label=config.model_names[model], alpha=0.7)
        else:
            plt.errorbar(x=x, y=df["loss"], yerr=df["loss_std"], c=config.model_colors[model], label=config.model_names[model], fmt="o", capsize=5, alpha=1.0)

    plt.xlabel(r"MMD bandwidth ($\sigma$)")
    plt.ylabel(r"MMD test loss")
    plt.yscale("log")
    plt.title(f"MMD Loss for {dataset_name}")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncols=2)
    plt.xticks(ticks=sigmas, labels=sigmas.round(2))
    plt.gca().set_facecolor('#F2F2F2')
    plt.grid(color='white', linestyle='--')
    plt.savefig(config.figures_folder / (dataset_name + "-mmd_loss.pdf"), dpi=500, bbox_inches="tight")
    plt.close()
