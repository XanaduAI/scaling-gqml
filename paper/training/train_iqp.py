import yaml
import iqpopt
import iqpopt.gen_qml as gen
from iqpopt.utils import *
from iqpopt import IqpSimulator, Trainer
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle

np.random.seed(666)

############### SETTINGS #################

bitflip = True  #to use the bitflip model or not
turbo = None
val_frac = None #fraction of the validation set. If None, convergence is decided on X_train

param_init_file = None #'trained_parameters/params_IqpSimulator_genomic-805.pkl'
#
# dataset_name = '8_blobs'
# dataset_path = '../datasets/blobs/8_blobs_dataset/16_spins_8_blobs_train.csv'

dataset_name = '2D_ising'
dataset_path = '../datasets/ising/2d_random_lattice_dataset/ising_4_4_T_3_train.csv'

# dataset_name = 'spin_glass'
# dataset_path = '../datasets/ising/spin_glass_dataset/ising_spin_glass_N_256_T_3.0_train.csv'

# dataset_name = 'dwave'
# dataset_path = '../datasets/dwave/dwave_X_train.csv'

# dataset_name = 'scale_free'
# dataset_path = '../datasets/ising/scale_free_dataset/ising_scale_free_1000_nodes_T_15_train.csv'

# dataset_name = 'MNIST'
# dataset_path = '../datasets/MNIST/x_train.csv'

# dataset_name = 'genomic-805'
# dataset_path = '../datasets/genomic/805_SNP_1000G_real_train.csv'

# dataset_name = 'genomic-10k'
# dataset_path = '../datasets/genomic/10K_SNP_1000G_real_train.csv'

#########################################

def prepare_iqp_training(hyperparams, X_train, param_init_file):
    """
    Create the necessary objects needed to train the iqp circuit
    Args:
        hyperparams (dict): dictionary of hyperparameter values
        X_train (array): train dataset
    """
    if hyperparams['gates_config']['name']=='gates_from_covariance':
        hyperparams['gates_config']['kwargs']['data'] =  X_train
    gates = globals()[hyperparams['gates_config']['name']](**hyperparams['gates_config']['kwargs'])

    model = IqpSimulator(gates=gates, **hyperparams['model_config'], bitflip=bitflip)
    trainer = Trainer(loss=gen.mmd_loss_iqp, **hyperparams['trainer_config'])
    train_config = hyperparams['train_config']

    if val_frac is not None:
        X_train, X_val = train_test_split(X_train, test_size=val_frac)

    loss_kwargs = hyperparams['loss_config']
    loss_kwargs['iqp_circuit'] = model
    loss_kwargs['ground_truth'] = X_train
    loss_kwargs['sqrt_loss'] = False

    if param_init_file is not None:
        with open(param_init_file, 'rb') as file:
            params = pickle.load(file)
            params_init = jnp.array(params)
    else:
        params_init = initialize_from_data(gates,
                                           X_train,
                                           scale=hyperparams['init_config']['init_scale'],
                                           param_noise=hyperparams['init_config']['param_noise'])

    loss_kwargs['params'] = params_init
    loss_kwargs['wires'] = list(range(X_train.shape[-1]))  # match output wires to dataset

    if val_frac is not None:
        val_kwargs = dict(loss_kwargs)
        del val_kwargs['params']
        val_kwargs['ground_truth'] = X_val
    else:
        val_kwargs = None

    return model, trainer, loss_kwargs, val_kwargs, train_config


with open('./best_hyperparameters.yaml', 'r') as file:
    hyperparams = yaml.safe_load(file)

X_train = pd.read_csv(dataset_path, delimiter=',', header=None).to_numpy()
if -1 in X_train:
    X_train = jnp.array((1 + X_train) // 2, dtype=int)

print(X_train)

name = 'IqpSimulator' if bitflip is False else 'IqpSimulatorBitflip'
Model, trainer, loss_kwargs, val_kwargs, train_config = \
    prepare_iqp_training(hyperparams[name][dataset_name], jnp.array(X_train), param_init_file)

print(Model.gates)

random_data = jnp.array(jax.random.bernoulli(jax.random.PRNGKey(43),
                                             shape=(1000,X_train.shape[-1])), dtype=int)

print('median heuristic: ' + str(gen.utils.median_heuristic(X_train[:1000])))
print('training sigmas: ' + str(loss_kwargs['sigma']))
print('num gates: ' + str(len(Model.gates)))
print('MMD^2 of random data (median): ' + str(gen.mmd_loss_samples(X_train[:1000], random_data, gen.utils.median_heuristic(X_train[:1000]))))

# train the model
trainer.train(**train_config, loss_kwargs=loss_kwargs, val_kwargs=val_kwargs, turbo=turbo,
              random_state=np.random.randint(0, 99999))

# save loss plots
sns.lineplot(trainer.losses, label='train')
if val_frac is not None:
    sns.lineplot(trainer.val_losses, label='validation')
plt.xlabel('training step')
plt.ylabel('MMD loss')
plt.yscale('log')
plt.savefig(f'./loss_plots/loss_plot_{name}_{dataset_name}_log.png', dpi=300)

plt.clf()
sns.lineplot(trainer.losses, label='train')
if val_frac is not None:
    sns.lineplot(trainer.val_losses, label='validation')
plt.xlabel('training step')
plt.ylabel('MMD loss')
plt.savefig(f'./loss_plots/loss_plot_{name}_{dataset_name}.png', dpi=300)

# save the losses
np.savetxt(f'./loss_plots/train_losses_{name}_{dataset_name}.csv', trainer.losses, delimiter=',')
if val_frac is not None:
    np.savetxt(f'./loss_plots/val_losses_{name}_{dataset_name}.csv', trainer.val_losses, delimiter=',')

# save the final parameters
with open(f'./trained_parameters/params_{name}_{dataset_name}.pkl', 'wb') as f:
    pickle.dump(trainer.final_params, f)
