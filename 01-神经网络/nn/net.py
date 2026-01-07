from nn import InputLayer,MiddleLayer,OutputLayer
from numpy.typing import NDArray

class Network:
    def __init__(self, network_shape: tuple) -> None:
        self.shape = network_shape
        self.layers = []
        self.layers.append(InputLayer(self.shape[0]))
        for node in range(0, len(self.shape) - 2):
            print(node)
            self.layers.append(MiddleLayer(self.shape[node],self.shape[node + 1]))
        self.layers.append(OutputLayer(self.shape[-2], self.shape[-1]))

    def net_forward(self) -> NDArray:
        inputs = self.layers[0].forward()
        for layer in self.layers[1:]:
            print(layer.weights)
            print(layer.biases)
            print(inputs)
            inputs = layer.forward(inputs)
        outputs = inputs
        print(outputs)
            