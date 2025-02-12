import rsmf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from iqpopt.gen_qml.utils import sigma_heuristic
from labellines import labelLine
from matplotlib.legend_handler import HandlerTuple
import pickle
import plots_config as config

############### SETTINGS #################

aspect_ratio = 0.6
show_sigmas = False

datasets = [
    # "8_blobs",
    # "2D_ising",
    # "spin_glass",
    # "scale_free",
    "MNIST",
    # "genomic-805",
    # "dwave",
]

#########################################

for dataset_name in datasets:
    dataset_path = config.test_paths[dataset_name]
    labels_path = config.test_label_paths.get(dataset_name, "")
    theres_labels = labels_path != ""

    X_test = pd.read_csv(dataset_path, delimiter=',', header=None).to_numpy(dtype=int)
    if -1 in X_test:
        X_test = np.array((1 + X_test) // 2, dtype=int)

    # plt_sigmas = np.array([np.sqrt(median_heuristic(X_test[:1000]))]).round(2)
    # plt_sigmas = np.array(sigma_heuristic(X_test[:1000], 3)).round(2)
    plt_sigmas = np.linspace(min(config.trained_sigmas[dataset_name]), max(config.trained_sigmas[dataset_name]), 3).round(2)
    

    with open(config.tables_folder / (dataset_name + "-kgel.pkl"), 'rb') as f:
        kgel_tbl = pickle.load(f)

    kgel_tbl = kgel_tbl.round({'sigma': 2})
    plt_sigmas = pd.Series(plt_sigmas, name='sigma')
    kgel_tbl = pd.merge(kgel_tbl, plt_sigmas, "inner")

    models = kgel_tbl["model"].unique()
    sigmas = kgel_tbl["sigma"].unique()

    if theres_labels:
        y_test = np.loadtxt(labels_path, dtype=int)

    kgel_tbl = kgel_tbl.dropna()
    kgel_not_null = kgel_tbl[kgel_tbl["pi"].notnull()]
    num_lines = len(kgel_not_null["pi"].values)
    len_x_axis = len(kgel_not_null["pi"].values[0])
    delta = len_x_axis/(num_lines+1)

    x = 0
    handles, labels = {}, {}
    for sigma in sigmas:
        
        formatter = rsmf.setup(config.formatter_setup)
        fig = formatter.figure(aspect_ratio=aspect_ratio)
        ax = fig.add_axes([0, 0, 1, 1])
        
        if theres_labels:
            num_bars = len(kgel_tbl[(kgel_tbl["pi"].notnull()) & (kgel_tbl["sigma"] == sigma)]["pi"].values)

        i = 0
        for model in models:
            df = kgel_tbl[(kgel_tbl["model"] == model) & (kgel_tbl["sigma"] == sigma)]

            if len(df["pi"].values) == 0:
                continue

            array = df["pi"].values[0]
            
            if array is not None:
                if not theres_labels:
                    h, = plt.plot(np.cumsum(np.sort(array)), c=config.model_colors[model], label=config.model_names[model])
                    handles[model] = tuple([h])
                    labels[model] = config.model_names[model]
                    x += delta
                    if show_sigmas:
                        labelLine(ax.get_lines()[-1], x, label=round(sigma, 2), align=False)
                
                else:
                    blobs_data = pd.DataFrame({"labels": y_test[:len_x_axis], "pi": array}).groupby(["labels"]).sum()
                    width = (blobs_data.index[-1] - blobs_data.index[0]) / (len(blobs_data.index)*(num_bars + 1) - 3)
                    plt.bar(blobs_data.index + width*i, blobs_data["pi"], color=config.model_colors[model], label=config.model_names[model] + " (" + str(round(df["kgel"].values[0], 2)) + ")", width=width)
                    plt.xticks(blobs_data.index + width/2 * (num_bars-1), blobs_data.index)
                    i += 1

        if theres_labels:
            plt.xlabel(r"Ground truth labels")
            plt.ylabel(r"Distribution")
            plt.axhline(1/len(blobs_data.index), c="black")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncols=2)
            plt.title(r"KGEL modes for $\sigma$ = " + str(round(sigma, 2)))
            plt.grid()
            plt.savefig(config.figures_folder / (dataset_name + f"-kgel-sigma_{round(sigma,2)}.pdf"), bbox_inches="tight", dpi=500)
            plt.close()

        else:
            plt.xlabel(r"Ground truth samples")
            plt.ylabel(r"Cumulative distribution")
            plt.title(r"KGEL distribution for $\sigma$ = " + str(round(sigma, 2)))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=False, ncols=2,
                    handles=list(handles.values()), labels=(labels.values()), handler_map={tuple: HandlerTuple(None)})
            plt.grid()
            plt.savefig(config.figures_folder / (dataset_name + f"-kgel-sigma_{round(sigma,2)}.pdf"), bbox_inches="tight", dpi=500)
            plt.close()
