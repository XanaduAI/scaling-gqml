import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import networkx as nx
from qml_benchmarks.models.energy_based_model import EnergyBasedModel
from qml_benchmarks.model_utils import mmd_loss, median_heuristic

class MaskedMLP(nn.Module):
    "Multilayer perceptron."
    # Create a MLP with a number of layers with the same number of features as the input
    # A mask is applied to weight matrices to add structure
    # The weight matrices are made symmetric to reflect the graph structure
    #
    n_layers: int
    mask: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        dim = x.shape[-1]
        for i in range(self.n_layers - 1):
            weights = self.param(f'weights_{i}', jax.nn.initializers.lecun_normal(), (dim, dim))
            weights = (weights + weights.T) / 2  # symmetrize weights
            weights = weights * self.mask  # apply mask
            bias = self.param(f'bias{i}', jax.nn.initializers.zeros, (dim,))
            x = jnp.dot(x, weights) + bias
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x

class DeepGraphEBM(EnergyBasedModel):
    """
    Energy-based model where the energy function is a neural network.
    The weights of the energy function reflect the structure of the graph.

    Args:
        n_layers (list[int]):
            The number of hidden layers
        G: networkx graph object

    """

    def __init__(self,
                 learning_rate=0.001,
                 batch_size=32,
                 max_steps=10000,
                 cdiv_steps=100,
                 convergence_interval=None,
                 random_state=42,
                 jit=True,
                 G=None,
                 n_layers=1,
                 mmd_kwargs={'n_samples': 1000, 'n_steps': 1000, 'sigma': 1.0}):
        super().__init__(
            dim=None,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
            cdiv_steps=cdiv_steps,
            convergence_interval=convergence_interval,
            random_state=random_state,
            jit=jit
        )

        if G is None:
            raise ValueError("must specify graph")

        self.n_layers = n_layers
        self.G = G
        self.adj_matrix = jnp.array(nx.adjacency_matrix(G).toarray())
        self.mmd_kwargs = mmd_kwargs
        self.model = None

    def initialize(self, x):
        dim = x.shape[1]
        if not isinstance(dim, int):
            raise NotImplementedError(
                "The model is not yet implemented for data"
                "with arbitrary dimensions. `dim` must be an integer."
            )

        self.dim = dim
        self.model = MaskedMLP(n_layers=self.n_layers, mask=self.adj_matrix)
        self.params_ = self.model.init(self.generate_key(), x)

    def energy(self, params, x):
        return self.model.apply(params, x)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        sigma = self.mmd_kwargs['sigma']
        sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
        score = np.mean([mmd_loss(X, self.sample(self.mmd_kwargs['n_samples'],
                                                 self.mmd_kwargs['n_steps']), sigma) for sigma in sigmas])
        return float(-score)


batched_matmul = jax.jit(jax.vmap(jax.numpy.matmul, in_axes=(0, 0)))

class GCNLayer(nn.Module):
    """
    code taken from
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial7/GNN_overview.html
    """
    c_out: int  # Output feature size

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Array with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(axis=-1, keepdims=True)
        node_feats = nn.Dense(features=self.c_out, name='projection')(node_feats)
        node_feats = batched_matmul(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        #         print(node_feats)
        #         print(node_feats.shape)
        return node_feats


class GCNModel(nn.Module):
    hidden_layers: int

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Array with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph, shape: [batch_size, num_nodes, num_nodes]
        Outputs:
            Single float prediction for each graph in the batch, shape: [batch_size, final_output_dim]
        """
        # Pass through multiple GCN layers
        for c_out in self.hidden_layers:
            node_feats = GCNLayer(c_out=c_out)(node_feats, adj_matrix)
            node_feats = nn.relu(node_feats)

        # Global average pooling over nodes (mean pooling across the nodes)
        graph_feats = jnp.mean(node_feats, axis=1)

        # Final dense layer for single output prediction
        output = nn.Dense(features=1)(graph_feats)
        return output


class GraphEBM(EnergyBasedModel):
    """
    Energy-based model with the energy function is a graph neural network.

    Args:
        hidden_layers (list[int]):
            A list that specifies the output dimension of each of the hidden layers in the network
    """

    def __init__(self,
                 learning_rate=0.001,
                 batch_size=32,
                 max_steps=10000,
                 cdiv_steps=100,
                 convergence_interval=None,
                 random_state=42,
                 jit=True,
                 hidden_layers=[8, 4],
                 adj_matrix=None,
                 mmd_kwargs={'n_samples': 1000, 'n_steps': 1000, 'sigma': 1.0}):
        super().__init__(
            dim=None,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
            cdiv_steps=cdiv_steps,
            convergence_interval=convergence_interval,
            random_state=random_state,
            jit=jit
        )

        if adj_matrix is None:
            raise ValueError("adjacency matrix must be specified")

        self.hidden_layers = hidden_layers
        self.adj_matrix = adj_matrix
        self.mmd_kwargs = mmd_kwargs

    def initialize(self, x):
        dim = x.shape[1]
        if not isinstance(dim, int):
            raise NotImplementedError(
                "The model is not yet implemented for data"
                "with arbitrary dimensions. `dim` must be an integer."
            )

        x = jnp.expand_dims(x, -1)
        self.dim = dim

        self.model = GCNModel(hidden_layers=self.hidden_layers)
        self.params_ = self.model.init(self.generate_key(), x[:1], jnp.array([self.adj_matrix]))

    def energy(self, params, x):
        x = jnp.expand_dims(x, -1)
        return self.model.apply(params, x, jnp.array([self.adj_matrix] * x.shape[0]))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        sigma = self.mmd_kwargs['sigma']
        sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
        score = np.mean([mmd_loss(X, self.sample(self.mmd_kwargs['n_samples'],
                                                 self.mmd_kwargs['n_steps']), sigma) for sigma in sigmas])
        return float(-score)


