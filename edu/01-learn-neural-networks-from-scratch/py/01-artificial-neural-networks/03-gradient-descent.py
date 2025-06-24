import numpy as np

# 输入数据
feature = np.array([28.1, 58.0])
# 实际结果
label = np.array([165])

# 模型参数（权重和偏置）
weight, bias = np.array([[1, 1]]) / 2, np.array([0])


# 前向传播函数
def forward(x, w, b):
    return x.dot(w.T) + b


# 反向传播函数
def backward(x, d, w, b):
    return w - d * x, b - np.sum(d)


# 损失函数（均方误差）
def mse_loss(p, y):
    return ((p - y) ** 2).mean()


# 梯度计算函数（损失函数的导数）
def gradient(p, y):
    return (p - y) * 2

# 模型推理
prediction = forward(feature, weight, bias)
# 模型验证
error = mse_loss(prediction, label)
# 梯度计算
delta = gradient(prediction, label)
# 反向传播
weight, bias = backward(feature, delta, weight, bias)

# 结果输出
print(f'预测冰淇淋销量：{prediction}')
print(f'均方误差：{error}')
print(f"权重：{weight}")
print(f"偏置：{bias}")