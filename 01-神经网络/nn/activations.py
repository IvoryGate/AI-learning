import numpy as np

# ReLu函数
def ReLu(input: float) -> float:
    return np.maximum(0, input)

def SoftMax(input: float) -> float:
    max_values = np.max(input, axis=1, keepdims=True)
    return input - max_values
