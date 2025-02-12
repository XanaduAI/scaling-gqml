from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import jax
import jax.numpy as jnp
from iqpopt.iqp_optimizer import IqpSimulator as IqpSimulatorCore
from iqpopt.training import *
from iqpopt.gen_qml import mmd_loss_iqp as mmd_loss
from iqpopt.gen_qml.utils import sigma_heuristic, median_heuristic
from iqpopt.utils import *
import gc

np.random.seed(666)


class IqpSimulator(BaseEstimator):
    """
    wrapped version of the model for use with sklearn hyperparameter opt
    """

    def __init__(self, gate_fn: str='local_gates', gate_arg1: int=0, gate_arg2: int=2, gate_arg3: int=2,  device: str="lightning.qubit",
                 n_ancilla: int=0, spin_sym: bool=False, init_gates: list=None, init_coefs: list=None, sparse: bool=False,
                 bitflip=False, loss: callable=mmd_loss, optimizer: str='Adam', stepsize: float=0.001,
                 opt_jit: bool=False, n_iters: int=1000, n_sigmas: int=1, train_sigmas=None, n_ops: int=100, n_samples: int=100, n_ops_score: int=1000,
                 n_samples_score: int=1000, n_repeats_score: int=5, score_sigmas: list=None, sqrt_loss: bool=False, convergence_interval: int or bool=None,  monitor_interval:int=None, turbo=None,
                 init_scale: float=1.0, param_noise: float=0., random_state=666) -> None:

        self.gate_fn = gate_fn
        self.gate_arg1 = gate_arg1
        self.gate_arg2 = gate_arg2
        self.gate_arg3 = gate_arg3
        self.device = device
        self.n_ancilla = n_ancilla
        self.spin_sym = spin_sym
        self.init_gates = init_gates
        self.init_coefs = init_coefs
        self.sparse = sparse
        self.bitflip = bitflip
        self.opt_jit = opt_jit
        self.loss = loss
        self.optimizer = optimizer
        self.stepsize = stepsize
        self.n_iters = n_iters
        self.n_sigmas = n_sigmas
        self.train_sigmas = train_sigmas
        self.n_ops = n_ops
        self.n_samples = n_samples
        self.n_ops_score = n_ops_score
        self.n_samples_score = n_samples_score
        self.n_repeats_score = n_repeats_score
        self.score_sigmas = score_sigmas
        self.sqrt_loss = sqrt_loss
        self.convergence_interval = convergence_interval
        self.monitor_interval = monitor_interval
        self.turbo = turbo
        self.init_scale = init_scale
        self.param_noise = param_noise
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.params_ = None
        self.model = None
        self.med = None
        self.loss_kwargs = None
        self.trainer = None

    def initialize(self, X: any = None):
        jax.clear_caches()
        n_qubits = X.shape[-1] + self.n_ancilla
        self.gate_arg2 = self.gate_arg2 if self.n_ancilla == 0 else 2  # avoid blowup of gates for ancillas
        self.train_sigmas = sigma_heuristic(X, self.n_sigmas) if self.train_sigmas is None else self.train_sigmas
        self.med = median_heuristic(X[:1000])
        self.trainer = Trainer(self.optimizer, self.loss, self.stepsize, self.opt_jit)
        if self.gate_fn == 'local_gates':
            gates = local_gates(n_qubits, self.gate_arg2)
        elif self.gate_fn == 'nearest_neighbour_gates':
            gates = nearest_neighbour_gates(self.gate_arg1, self.gate_arg2, self.gate_arg3)
        else:
            print('gate fn not implemented')
        self.model = IqpSimulatorCore(n_qubits, gates, self.device, self.spin_sym, self.init_gates,
                                  self.init_coefs, self.sparse, self.bitflip)
        self.params_ = initialize_from_data(gates, jnp.hstack((X, jnp.zeros((X.shape[0], self.n_ancilla)))),
                                            scale=self.init_scale, param_noise=self.param_noise)

    def fit(self, X: jnp.array, y: any = None) -> None:
        """
        Fit the parameters and update self.params_.
        """
        self.initialize(X)
        self.loss_kwargs = {'params': self.params_, 'sigma': self.train_sigmas, 'n_ops': self.n_ops,
                            'n_samples': self.n_samples,
                            'key': jax.random.PRNGKey(self.rng.integers(0, 999999)), 'iqp_circuit': self.model,
                            'ground_truth': X, 'wires': list(range(X.shape[-1])), 'sqrt_loss': self.sqrt_loss}
        self.trainer.train(self.n_iters, self.loss_kwargs, None, self.convergence_interval,
                           self.monitor_interval, self.turbo, self.rng.integers(999999))
        self.params_ = self.trainer.final_params
        jax.clear_caches()
        gc.collect()

    def score(self, X, y=None):
        """Score the model on the given data.
        lower is better.
        """
        score_kwargs = dict(self.loss_kwargs)
        score_kwargs.pop('params')
        score_kwargs['ground_truth'] = X
        score_kwargs['sigma'] = self.score_sigmas
        score_kwargs['n_ops'] = self.n_ops_score
        score_kwargs['n_samples'] = self.n_samples_score

        scores = []
        for __ in range(self.n_repeats_score):
            score_kwargs['key'] = jax.random.PRNGKey(self.rng.integers(0, 999999))
            scores.append(float(self.loss(self.params_, **score_kwargs, indep_estimates=False)))

        return float(-np.mean(scores))

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
