# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

hyper_parameter_settings = {
    "IqpSimulator": {
        "2D_ising": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["local_gates"]},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [2, 4, 6]},
            "n_ancilla": {
                "type": "list",
                "dtype": "int",
                "val": [0, 8]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01]},
            "n_sigmas": {
                "type": "list",
                "dtype": "float",
                "val": [1]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "convergence_interval": {
                "type": "list",
                "dtype": "int",
                "val": [200]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [10000]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.01, 1.0, -1]},
            "param_noise": {
                "type": "list",
                "dtype": "float",
                "val": [0., 0.0001]},
            "spin_sym": {
                "type": "list",
                "dtype": "bool",
                "val": [True]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [1]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
           },
        "8_blobs": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["local_gates"]},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [2, 4, 6]},
            "n_ancilla": {
                "type": "list",
                "dtype": "int",
                "val": [0, 8]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01]},
            "n_sigmas": {
                "type": "list",
                "dtype": "float",
                "val": [1]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "convergence_interval": {
                "type": "list",
                "dtype": "int",
                "val": [200]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [10000]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.01, 1.0, -1]},
            "param_noise": {
                "type": "list",
                "dtype": "float",
                "val": [0., 0.0001]},
            "spin_sym": {
                "type": "list",
                "dtype": "bool",
                "val": [False]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [1]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
           },
        "spin_glass": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["local_gates"]},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [2]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01, 0.1]},
            "n_sigmas": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "convergence_interval": {
                "type": "list",
                "dtype": "int",
                "val": [500]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [3000]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01, 1, -1]},
            "spin_sym": {
                "type": "list",
                "dtype": "bool",
                "val": [True, False]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [5]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "turbo": {
                "type": "list",
                "dtype": "int",
                "val": [100]},
           },
        "dwave": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["local_gates"]},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [2]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01]},
            "n_sigmas": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "convergence_interval": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01, -1]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [5]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "turbo": {
                "type": "list",
                "dtype": "int",
                "val": [10]},
           },
        "MNIST": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["local_gates"]},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [2]},
            "n_ancilla": {
                "type": "list",
                "dtype": "int",
                "val": [0]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001]},
            "n_sigmas": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [400]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, -1]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "sparse": {
                "type": "list",
                "dtype": "bool",
                "val": [True]},
           },
        "scale_free": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["nearest_neighbour_gates"]},
            "gate_arg1": {
                "type": "list",
                "dtype": "str",
                "val": ['../datasets/ising/scale_free_dataset/graph.adjlist']},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [1, 2]},
            "gate_arg3": {
                "type": "list",
                "dtype": "int",
                "val": [2]},
            "n_ancilla": {
                "type": "list",
                "dtype": "int",
                "val": [0]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01]},
            "n_sigmas": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [2000]},
            "convergence_interval": {
                "type": "list",
                "dtype": "int",
                "val": [200]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01, -1]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "sparse": {
                "type": "list",
                "dtype": "bool",
                "val": [False]},
            "turbo": {
                "type": "list",
                "dtype": "int",
                "val": [100]},
           },
        "genomic-805": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["local_gates"]},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [2]},
            "n_ancilla": {
                "type": "list",
                "dtype": "int",
                "val": [0]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001]},
            "n_sigmas": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [400]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.1, -1]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "sparse": {
                "type": "list",
                "dtype": "bool",
                "val": [True]},
           },
        },
    "IqpSimulatorBitflip": {
        "2D_ising": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["local_gates"]},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [2, 4, 6]},
            "n_ancilla": {
                "type": "list",
                "dtype": "int",
                "val": [0, 8]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01]},
            "n_sigmas": {
                "type": "list",
                "dtype": "float",
                "val": [1]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "convergence_interval": {
                "type": "list",
                "dtype": "int",
                "val": [200]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [10000]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.01, 1.0, -1]},
            "param_noise": {
                "type": "list",
                "dtype": "float",
                "val": [0., 0.0001]},
            "spin_sym": {
                "type": "list",
                "dtype": "bool",
                "val": [True]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [1]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
           },
        "8_blobs": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["local_gates"]},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [2, 4, 6]},
            "n_ancilla": {
                "type": "list",
                "dtype": "int",
                "val": [0]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01]},
            "n_sigmas": {
                "type": "list",
                "dtype": "float",
                "val": [1]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "convergence_interval": {
                "type": "list",
                "dtype": "int",
                "val": [200]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [10000]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.01, 1.0, -1]},
            "param_noise": {
                "type": "list",
                "dtype": "float",
                "val": [0., 0.0001]},
            "spin_sym": {
                "type": "list",
                "dtype": "bool",
                "val": [False]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [1]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
           },
        "spin_glass": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["local_gates"]},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [2]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01, 0.1]},
            "n_sigmas": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "convergence_interval": {
                "type": "list",
                "dtype": "int",
                "val": [500]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [3000]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01, 1, -1]},
            "spin_sym": {
                "type": "list",
                "dtype": "bool",
                "val": [True, False]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [5]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "turbo": {
                "type": "list",
                "dtype": "int",
                "val": [100]},
           },
        "dwave": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["local_gates"]},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [2]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01]},
            "n_sigmas": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "convergence_interval": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01, -1]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [5]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "turbo": {
                "type": "list",
                "dtype": "int",
                "val": [10]},
           },
        "MNIST": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["local_gates"]},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [2]},
            "n_ancilla": {
                "type": "list",
                "dtype": "int",
                "val": [0]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001]},
            "n_sigmas": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [400]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, -1]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "sparse": {
                "type": "list",
                "dtype": "bool",
                "val": [True]},
           },
        "scale_free": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["nearest_neighbour_gates"]},
            "gate_arg1": {
                "type": "list",
                "dtype": "str",
                "val": ['../datasets/ising/scale_free_dataset/graph.adjlist']},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [1]},
            "gate_arg3": {
                "type": "list",
                "dtype": "int",
                "val": [2, 3]},
            "n_ancilla": {
                "type": "list",
                "dtype": "int",
                "val": [0]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01]},
            "n_sigmas": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [2000]},
            "convergence_interval": {
                "type": "list",
                "dtype": "int",
                "val": [200]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.01, -1]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "sparse": {
                "type": "list",
                "dtype": "bool",
                "val": [False]},
            "turbo": {
                "type": "list",
                "dtype": "int",
                "val": [100]},
           },
        "genomic-805": {
            "gate_fn": {
                "type": "list",
                "dtype": "str",
                "val": ["local_gates"]},
            "gate_arg2": {
                "type": "list",
                "dtype": "int",
                "val": [2]},
            "n_ancilla": {
                "type": "list",
                "dtype": "int",
                "val": [0]},
            "stepsize": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001]},
            "n_sigmas": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_iters": {
                "type": "list",
                "dtype": "int",
                "val": [400]},
            "init_scale": {
                "type": "list",
                "dtype": "float",
                "val": [0.0001, 0.001, 0.1, -1]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "n_ops_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "n_samples_score": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "sparse": {
                "type": "list",
                "dtype": "bool",
                "val": [True]},
           },
    },
    "RestrictedBoltzmannMachine": {
        "2D_ising": {
            "n_components": {
                "type": "list",
                "dtype": "int",
                "val": [4, 16, 64, 128, 256]},
            "learning_rate": {
                "type": "list",
                "dtype": "float",
                "val": [0.00001, 0.0001, 0.001, 0.01]},
            "batch_size": {
                "type": "list",
                "dtype": "int",
                "val": [16, 32, 64]},
            "n_iter": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "score_fn": {
                "type": "list",
                "dtype": "str",
                "val": ['mmd']},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [1]},
            "verbose": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
           },
        "8_blobs": {
            "n_components": {
                "type": "list",
                "dtype": "int",
                "val": [4, 16, 64, 128, 256]},
            "learning_rate": {
                "type": "list",
                "dtype": "float",
                "val": [0.00001, 0.0001, 0.001, 0.01]},
            "batch_size": {
                "type": "list",
                "dtype": "int",
                "val": [16, 32, 64]},
            "n_iter": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "score_fn": {
                "type": "list",
                "dtype": "str",
                "val": ['mmd']},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [1]},
            "verbose": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
           },
        "spin_glass": {
            "n_components": {
                "type": "list",
                "dtype": "int",
                "val": [128, 256, 512, 1024]},
            "learning_rate": {
                "type": "list",
                "dtype": "float",
                "val": [0.00001, 0.0001, 0.001]},
            "batch_size": {
                "type": "list",
                "dtype": "int",
                "val": [16, 32, 64]},
            "n_iter": {
                "type": "list",
                "dtype": "int",
                "val": [10000]},
            "score_fn": {
                "type": "list",
                "dtype": "str",
                "val": ['mmd']},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "verbose": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
           },
        "dwave": {
            "n_components": {
                "type": "list",
                "dtype": "int",
                "val": [128, 484, 1024]},
            "learning_rate": {
                "type": "list",
                "dtype": "float",
                "val": [0.00001, 0.0001, 0.001]},
            "batch_size": {
                "type": "list",
                "dtype": "int",
                "val": [16, 32, 64]},
            "n_iter": {
                "type": "list",
                "dtype": "int",
                "val": [300]},
            "score_fn": {
                "type": "list",
                "dtype": "str",
                "val": ['mmd']},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "verbose": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
           },
        "scale_free": {
            "n_components": {
                "type": "list",
                "dtype": "int",
                "val": [100, 250, 500]},
            "learning_rate": {
                "type": "list",
                "dtype": "float",
                "val": [0.00001, 0.0001, 0.001]},
            "batch_size": {
                "type": "list",
                "dtype": "int",
                "val": [32, 64]},
            "n_iter": {
                "type": "list",
                "dtype": "int",
                "val": [1000]},
            "score_fn": {
                "type": "list",
                "dtype": "str",
                "val": ['mmd']},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
            "verbose": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
           },
        "MNIST": {
            "n_components": {
                "type": "list",
                "dtype": "int",
                "val": [250]},
            "learning_rate": {
                "type": "list",
                "dtype": "float",
                "val": [0.00001, 0.0001, 0.001, 0.01]},
            "batch_size": {
                "type": "list",
                "dtype": "int",
                "val": [16, 32, 64]},
            "n_iter": {
                "type": "list",
                "dtype": "int",
                "val": [200]},
            "score_fn": {
                "type": "list",
                "dtype": "str",
                "val": ['mmd']},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
           },
    },
    "DeepEBM": {
        "2D_ising": {
            "hidden_layers": {
                "type": "list",
                "dtype": "tuple",
                "val": ["(16,)", "(16, 4, 2)", "(32, 32, 32)", "(100,)"]},
            "learning_rate": {
                "type": "list",
                "dtype": "float",
                "val": [0.00001, 0.0001, 0.001]},
            "batch_size": {
                "type": "list",
                "dtype": "int",
                "val": [16, 32]},
            "max_steps": {
                "type": "list",
                "dtype": "int",
                "val": [50000]},
            "cdiv_steps": {
                "type": "list",
                "dtype": "int",
                "val": [1, 10]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [1]},
           },
        "8_blobs": {
            "hidden_layers": {
                "type": "list",
                "dtype": "tuple",
                "val": ["(16,)", "(16, 4, 2)", "(32, 32, 32)", "(100,)"]},
            "learning_rate": {
                "type": "list",
                "dtype": "float",
                "val": [0.00001, 0.0001, 0.001]},
            "batch_size": {
                "type": "list",
                "dtype": "int",
                "val": [16, 32]},
            "max_steps": {
                "type": "list",
                "dtype": "int",
                "val": [50000]},
            "cdiv_steps": {
                "type": "list",
                "dtype": "int",
                "val": [1, 10]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [1]},
           },
        "dwave": {
            "hidden_layers": {
                "type": "list",
                "dtype": "tuple",
                "val": ["(484,)", "(484, 242, 100)", "(256, 256)", "(1000,)"]},
            "learning_rate": {
                "type": "list",
                "dtype": "float",
                "val": [0.00001, 0.0001, 0.001]},
            "batch_size": {
                "type": "list",
                "dtype": "int",
                "val": [64, 128]},
            "max_steps": {
                "type": "list",
                "dtype": "int",
                "val": [50000]},
            "cdiv_steps": {
                "type": "list",
                "dtype": "int",
                "val": [1, 10]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
           },
        "spin_glass": {
            "hidden_layers": {
                "type": "list",
                "dtype": "tuple",
                "val": ["(256,)", "(256, 128, 64, 32)", "(256, 256)", "(5000,)"]},
            "learning_rate": {
                "type": "list",
                "dtype": "float",
                "val": [0.00001, 0.0001, 0.001]},
            "batch_size": {
                "type": "list",
                "dtype": "int",
                "val": [64, 128]},
            "max_steps": {
                "type": "list",
                "dtype": "int",
                "val": [100000]},
            "cdiv_steps": {
                "type": "list",
                "dtype": "int",
                "val": [1, 10]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
           },
    },
    "DeepGraphEBM": {
        "scale_free": {
            "n_layers": {
                "type": "list",
                "dtype": "int",
                "val": [1, 2]},
            "learning_rate": {
                "type": "list",
                "dtype": "float",
                "val": [0.00001, 0.0001, 0.001]},
            "batch_size": {
                "type": "list",
                "dtype": "int",
                "val": [64]},
            "max_steps": {
                "type": "list",
                "dtype": "int",
                "val": [300000]},
            "cdiv_steps": {
                "type": "list",
                "dtype": "int",
                "val": [1, 10]},
            "n_sigmas_score": {
                "type": "list",
                "dtype": "int",
                "val": [3]},
           },
    }
}
