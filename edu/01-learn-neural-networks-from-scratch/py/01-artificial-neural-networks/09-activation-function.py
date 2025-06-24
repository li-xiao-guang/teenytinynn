import numpy as np

np.random.seed(99)

# 输入数据
features = np.array([[28.1, 58.0],
                     [22.5, 72.0],
                     [31.4, 45.0],
                     [19.8, 85.0],
                     [27.6, 63]])
# 实际结果
labels = np.array([[165],
                   [95],
                   [210],
                   [70],
                   [155]])

# 模型参数（隐藏层权重和偏置）
# 隐藏层
h_weight, h_bias = np.random.rand(4, 2) / 2, np.zeros(4)
# 输出层
o_weight, o_bias = np.random.rand(1, 4) / 4, np.zeros(1)


# 前向传播函数
def forward(x, w, b):
    return x.dot(w.T) + b


# 反向传播函数
def backward(x, d, w, b, lr):
    return w - d.T.dot(x) * lr, b - np.sum(d, axis=0) * lr


# 激活函数
def relu(x):
    return np.maximum(0, x)


# 激活函数反向传播函数
def relu_backward(y, d):
    return (y > 0) * d


# 损失函数（均方误差）
def mse_loss(p, y):
    return ((p - y) ** 2).mean(axis=0)


# 梯度计算函数（损失函数的导数）
def gradient(p, y):
    return (p - y) * 2


# 梯度计算函数反向传播函数
def gradient_backward(d, w):
    return d.dot(w)


# 学习率
LEARNING_RATE = 0.00001
# 轮次
EPOCHES = 1000
# 批次
BATCHES = 2

# 轮次
for epoch in range(EPOCHES):
    # 迭代
    error = 0
    for i in range(0, len(features), BATCHES):
        # 批次
        feature, label = features[i: i + BATCHES], labels[i: i + BATCHES]

        # 模型推理
        hidden = relu(forward(feature, h_weight, h_bias))
        prediction = forward(hidden, o_weight, o_bias)
        # 模型验证
        error += mse_loss(prediction, label) * len(feature)
        # 梯度计算
        o_delta = gradient(prediction, label)
        h_delta = relu_backward(hidden, gradient_backward(o_delta, o_weight))
        # 反向传播
        o_weight, o_bias = backward(hidden, o_delta, o_weight, o_bias, LEARNING_RATE)
        h_weight, h_bias = backward(feature, h_delta, h_weight, h_bias, LEARNING_RATE)

    # 结果输出
    print(f"训练周期：{epoch}")
    print(f'预测冰淇淋销量：{prediction}')
    print(f'均方误差：{error / len(features)}')
    print(f"隐藏层权重：{h_weight}")
    print(f"隐藏层偏置：{h_bias}")
    print(f"输出层权重：{o_weight}")
    print(f"输出层偏置：{o_bias}")
