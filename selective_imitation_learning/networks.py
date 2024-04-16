from typing import List

import equinox as eqx
import jax

from selective_imitation_learning.utils import to_simplex


class MLP(eqx.Module):
    layers: List

    def __init__(self, key, in_size, out_size, hidden_size=128, num_hidden_layers=1):
        self.layers, keys = [], jax.random.split(key, num_hidden_layers + 2)
        self.layers.append(eqx.nn.Linear(in_size, hidden_size, key=keys[0]))
        for i in range(1, num_hidden_layers + 1):
            self.layers.append(eqx.nn.Linear(hidden_size, hidden_size, key=keys[i]))
        self.layers.append(eqx.nn.Linear(hidden_size, out_size, key=keys[-1]))

    @jax.jit
    def __call__(self, x):
        x = x.flatten()
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        out = self.layers[-1](x)
        return out / (out.sum() + 1e-6)
