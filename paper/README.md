# How to load and sample trained models

```python
import sys
sys.path.append("paper")
from qml_benchmarks.models import DeepEBM, RestrictedBoltzmannMachine
from flax_utils import DeepGraphEBM
from iqpopt import *
from iqpopt.utils import *
import numpy as np
import yaml
import pickle

with open('paper/training/best_hyperparameters.yaml', 'r') as f:
    best_hyperparams = yaml.safe_load(f)

X = np.loadtxt('paper/datasets/ising/2d_random_lattice_dataset/ising_4_4_T_3_train.csv', delimiter=',', dtype='int32')
```

## RBM
Load model directly with pickle
```python
with open('paper/training/trained_parameters/params_RestrictedBoltzmannMachine_2D_ising.pkl', 'rb') as f:
    rbm_model = pickle.load(f)
sample = rbm_model.sample(100)
```

## DeepEBM
Load using qml benchmarks package
```python
with open('paper/training/trained_parameters/params_DeepEBM_2D_ising.pkl', 'rb') as f:
    params_ebm = pickle.load(f)

ebm_model = DeepEBM(**best_hyperparams['DeepEBM']['2D_ising'])
ebm_model.initialize(X)
ebm_model.params_ = params_ebm #set the parameters
ebm_model.sample(100)
```

## DeepGraphEBM
```python
with open('paper/training/trained_parameters/params_DeepGraphEBM_scale_free.pkl', 'rb') as f:
    params_gebm = pickle.load(f)

X = np.loadtxt('paper/datasets/ising/scale_free_dataset/ising_scale_free_1000_nodes_T_1_train.csv', delimiter=',', dtype='int32')
G = nx.read_adjlist('paper/datasets/ising/scale_free_dataset/graph.adjlist')

ebm_model = DeepGraphEBM(G=G, **best_hyperparams['DeepGraphEBM']['scale_free'])
ebm_model.initialize(X)
ebm_model.params_ = params_gebm #set the parameters
ebm_model.sample(100)
```



## IqpSimulator
```python
with open('paper/training/trained_parameters/params_IqpSimulator_2D_ising.pkl', 'rb') as f:
    params_iqp = pickle.load(f)

gate_fn = globals()[best_hyperparams['IqpSimulator']['2D_ising']['gates_config']['name']]
gates = gate_fn(**best_hyperparams['IqpSimulator']['2D_ising']['gates_config']['kwargs'])

model = IqpSimulator(**best_hyperparams['IqpSimulator']['2D_ising']['model_config'], gates=gates)
model.sample(params_iqp, 100)
```