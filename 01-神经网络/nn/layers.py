from tools import create_inputs, create_weights, create_biases
from numpy.typing import NDArray
from nn import ReLu
import numpy as np

class Layer:
    def __init__(self, input_n: int, output_n: int) -> None:
        self.input_n = input_n
        self.output_n = output_n
        self.weights = create_weights(inputs_n=self.input_n, neurons_n=self.output_n)
        self.biases = create_biases(self.output_n)
    
class InputLayer():
    def __init__(self, input_n: int) -> None:
        self.input_n = input_n
    
    def forward(self) -> None:
        self.inputs = create_inputs(self.input_n)
        return self.inputs

class MiddleLayer(Layer):
    def __init__(self, input_n: int, output_n: int) -> None:
        super().__init__(input_n, output_n)
    
    def forward(self, inputs):
        return ReLu(np.dot(inputs, self.weights) + self.biases)

class OutputLayer(Layer):
    def __init__(self, input_n: int, output_n: int) -> None:
        super().__init__(input_n, output_n)
    
    def forward(self, inputs):
        return ReLu(np.dot(inputs, self.weights) + self.biases)