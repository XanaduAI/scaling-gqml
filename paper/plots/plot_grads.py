import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pickle
import rsmf
import plots_config as config
from matplotlib.ticker import ScalarFormatter


dataset_names = [
            '2D_ising',
            '8_blobs',
            'dwave',
            'MNIST',
            'scale_free',
            'genomic-805'
            ]

title_names = [
            '2D Ising',
            'blobs',
            'D-Wave',
            'MNIST',
            'scale free',
            'genomic'
            ]

with open("./grads/grads_cov.pkl", "rb") as f:
    grads_cov = pickle.load(f)

with open("./grads/grads_rand.pkl", "rb") as f:
    grads_rand = pickle.load(f)


fig, axes = plt.subplots(ncols=len(dataset_names), nrows=2, figsize=(13, 5))
plt.tight_layout()

plt.subplots_adjust(hspace=0.3)

formatter = ScalarFormatter(useMathText=True)  # Use MathText for scientific notation
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 3))  # Force scientific notation within this range

formatter_y = ScalarFormatter(useMathText=True)  # Use MathText for scientific notation
formatter_y.set_scientific(True)
formatter_y.set_powerlimits((-3, 3))  # Force scientific notation within this range

for i, dataset_name in enumerate(dataset_names):
    data_cov = jnp.abs(np.array(np.mean(grads_cov[dataset_name], axis=0)))
    data_rand = jnp.abs(np.array(np.mean(grads_rand[dataset_name], axis=0)))
    bins = np.linspace(min(data_cov.min(), data_rand.min()), max(data_cov.max(), data_rand.max()), 50)
    axes[0, i].xaxis.set_major_formatter(formatter)
    # axes[0, i].yaxis.set_major_formatter(formatter_y)
    axes[0, i].hist(data_cov, bins=bins, label='data init', alpha=0.7, color=config.model_colors['IqpSimulator'])
    axes[0, i].hist(data_rand, bins=bins, label='random init', alpha=0.7, color=config.model_colors['DeepEBM'])
    axes[0, i].set_yscale('log')
    axes[0, i].set_facecolor('#f0f0f0')
    if i == 5:
        axes[0, i].legend()
    axes[0, i].set_title(title_names[i])



for i, dataset_name in enumerate(dataset_names):
    axes[1, i].yaxis.set_major_formatter(formatter_y)
    # axes[1, i].xaxis.set_major_formatter(formatter)
    data_cov = jnp.abs(np.array(np.mean(grads_cov[dataset_name], axis=0)))
    data_rand = jnp.abs(np.array(np.mean(grads_rand[dataset_name], axis=0)))
    bins = np.linspace(min(data_cov.min(), data_rand.min()), max(0., data_rand.max()), 50)
    axes[1, i].hist(data_cov, bins=bins, label='data init', alpha=0.7, color=config.model_colors['IqpSimulator'])
    axes[1, i].hist(data_rand, bins=bins, label='random init', alpha=0.7, color=config.model_colors['DeepEBM'])
    #     axes[1,i].set_yscale('log')
    if i == 5:
        axes[1, i].legend()
    #     axes[1,i].set_title(title_names[i])
    # axes[1, i].set_xlabel('gradient magnitude', labelpad=20)
    axes[1, i].set_facecolor('#f0f0f0')



print('gradient component std means:' +str([np.mean(np.std(grads_cov[dataset_name], axis=0)) for dataset_name in dataset_names]))


plt.savefig('./figures/grads.pdf', dpi=300, bbox_inches='tight')