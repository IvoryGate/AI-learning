# 手动实现简单神经网络

## 前期准备

1. 创建python虚拟环境
2. 安装依赖（仅使用numpy，见`requirements`）

```bash
# 创建虚拟环境
python3 -m venv .myvenv
# 激活虚拟环境
## Windows
.myvenv\Scripts\activate
## Linux/macOS
source .myvenv/bin/activate
# 安装依赖
pip install -r requirements.txt
```
## 开始

### 激活函数

```python
# ReLu函数
def ReLu(input: float) -> float:
    return np.maximum(0, input)
```

### 随机生成输入，权重，偏置值

```python
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
```

```python
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
```

### 实现层类

```python
class Layer:
    def __init__(self, input_n: int, output_n: int) -> None:
        self.input_n = input_n
        self.output_n = output_n
        self.weights = create_weights(inputs_n=self.input_n, neurons_n=self.output_n)
        self.biases = create_biases(self.output_n)
    
class InputLayer():
    def __init__(self, input_n: int) -> None:
        self.input_n = input_n
        self.create_inputs()
    
    def create_inputs(self) -> None:
        self.inputs = create_inputs(self.input_n)

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
```
执行一下

```python
    input_layer = InputLayer(2)
    middle_layer1 = MiddleLayer(2, 3)
    middle_layer2 = MiddleLayer(3, 4)
    output_layer = OutputLayer(4, 2)
    print(output_layer.forward(middle_layer2.forward(middle_layer1.forward(input_layer.inputs))))
```

### 实现网络类