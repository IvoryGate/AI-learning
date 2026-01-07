import numpy as np
from nn import ReLu
from tools import create_inputs, create_weights, create_biases
from nn import InputLayer, MiddleLayer, OutputLayer

if __name__ == "__main__":
    # make a four layers net

    # input layer
    inputs = create_inputs(2)

    # layer 1
    weights1 = create_weights(2, 3)
    biases1 = create_biases(3)

    # layer 2
    weights2 = create_weights(3, 4)
    biases2 = create_biases(4)

    # out layer
    weights3 = create_weights(4 ,2)
    biases3 = create_biases(2)

    # output
    output1 = ReLu(np.dot(inputs, weights1) + biases1)
    output2 = ReLu(np.dot(output1, weights2) + biases2)
    output3 = ReLu(np.dot(output2, weights3) + biases3)

    print(output3)

    input_layer = InputLayer(2)
    middle_layer1 = MiddleLayer(2, 3)
    middle_layer2 = MiddleLayer(3, 4)
    output_layer = OutputLayer(4, 2)
    print(
        output_layer.forward(
            middle_layer2.forward(
                middle_layer1.forward(
                    input_layer.inputs
                )
            )
        )
    )

