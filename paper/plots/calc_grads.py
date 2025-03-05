import numpy as np
import matplotlib.pyplot as plt
import yaml
import jax
import jax.numpy as jnp
import pickle
import iqpopt.gen_qml as gen
from iqpopt.utils import *
from iqpopt import IqpSimulator, Trainer
from sklearn.model_selection import train_test_split

n_samples = 1000
n_ops = 1000
max_batch_samples = 500
max_batch_ops = 500
M = 1000

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


datasets = [
            '../datasets/ising/2d_random_lattice_dataset/ising_4_4_T_3_train.csv',
            '../datasets/blobs/8_blobs_dataset/16_spins_8_blobs_train.csv',
            # '../datasets/dwave/dwave_X_train.csv',
            # '../datasets/MNIST/x_train.csv',
            # '../datasets/ising/scale_free_dataset/ising_scale_free_1000_nodes_T_1_train.csv',
            # '../datasets/genomic/805_SNP_1000G_real_train.csv'
            ]

dataset_names = [
            '2D_ising',
            '8_blobs',
            # 'dwave',
            # 'MNIST',
            # 'scale_free',
            # 'genomic-805'
            ]

bitflip = False
val_frac = None

n_trials = 10
grads_cov = {}
grads_rand = {}

with open('../training/best_hyperparameters.yaml', 'r') as f:
    hyperparams = yaml.safe_load(f)

for dataset, dataset_name in zip(datasets[:], dataset_names[:]):
    print(dataset_name)
    grads_cov[dataset_name] = []
    grads_rand[dataset_name] = []
    X_train = np.loadtxt(dataset, delimiter=',')
    if dataset_name in ['MNIST', 'scale_free', 'genomic-805']:
        hyperparams['IqpSimulator'][dataset_name]['sparse'] = True
    Model, trainer, loss_kwargs, val_kwargs, train_config = \
        prepare_iqp_training(hyperparams['IqpSimulator'][dataset_name], jnp.array(X_train), None)
    loss_kwargs['n_samples'] = n_samples
    loss_kwargs['n_ops'] = n_ops

    params = loss_kwargs.pop('params')
    params_rand = jnp.array(np.random.rand(*params.shape)) * 2 * jnp.pi
    for __ in range(n_trials):
        grads = jax.grad(gen.mmd_loss_iqp, argnums=0)(params, **loss_kwargs,
                                                      key=jax.random.PRNGKey(np.random.randint(999999)),
                                                      max_batch_ops=max_batch_ops,
                                                      max_batch_samples=max_batch_samples,
                                                      jit=False)

        grads_cov[dataset_name].append(jnp.array(grads))

        grads = jax.grad(gen.mmd_loss_iqp, argnums=0)(params_rand, **loss_kwargs,
                                                      key=jax.random.PRNGKey(np.random.randint(999999)),
                                                      max_batch_ops=max_batch_ops,
                                                      max_batch_samples=max_batch_samples,
                                                      jit=False)

        grads_rand[dataset_name].append(jnp.array(grads))

with open("./grads/grads_cov.pkl", "wb") as f:
    pickle.dump(grads_cov, f)

with open("./grads/grads_rand.pkl", "wb") as f:
    pickle.dump(grads_rand, f)



