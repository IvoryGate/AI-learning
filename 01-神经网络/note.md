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