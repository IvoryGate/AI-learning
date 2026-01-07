import numpy as np
from numpy.typing import NDArray

def create_inputs(
        inputs_n: int,
    ) -> NDArray:
        return np.random.randn(inputs_n)

def create_weights(
        inputs_n: int,
        neurons_n: int
    ) -> NDArray:
        return np.random.randn(inputs_n, neurons_n)

def create_biases(
        neurons_n: int
    ) -> NDArray:
        return np.random.randn(neurons_n)