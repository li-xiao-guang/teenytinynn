import numpy as np

# 输入数据
feature = np.array([28.1, 58.0])
# 实际结果
label = np.array([165])

# 模型参数（权重和偏置）
weight, bias = np.array([[1, 1]]) / 2, np.array([0])


# 神经元逻辑（线性回归（多元一次）函数）
def forward(x, w, b):
    return x.dot(w.T) + b


# 反向传播
def backward(x, d, w, b, lr):
    return w - d * x * lr, b - np.sum(d) * lr


# 损失函数（均方差）
def mse_loss(p, y):
    return ((p - y) ** 2).mean()


# 梯度计算（损失函数的导数）
def gradient(p, y):
    return (p - y) * 2


# 学习率
LEARNING_RATE = 0.00001

# 模型推理
prediction = forward(feature, weight, bias)
# 模型验证
error = mse_loss(prediction, label)
# 梯度计算
delta = gradient(prediction, label)
# 反向传播
weight, bias = backward(feature, delta, weight, bias, LEARNING_RATE)
# 结果输出
print(f'预测冰淇淋销量：{prediction}')
print(f'均方差：{error}')
print(f"权重：{weight}")
print(f"偏置：{bias}")
