import numpy as np
import numpy.typing as npt

def create_inputs(
        inputs_n: int,
    ) -> npt:
        return np.random.randn(inputs_n)

def create_weights(
        inputs_n: int,
        neurons_n: int
    ) -> npt:
        return np.random.randn(inputs_n, neurons_n)

def create_biases(
        neurons_n: int
    ) -> npt:
        return np.random.randn(neurons_n)