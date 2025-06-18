import numpy as np

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

# 模型参数（权重和偏置）
weight, bias = np.array([[1, 1]]) / 2, np.array([0])


# 神经元逻辑（线性回归（多元一次）函数）
def forward(x, w, b):
    return x.dot(w.T) + b


# 损失函数（均方差）
def mse_loss(p, y):
    return ((p - y) ** 2).mean()


# 迭代
for i in range(len(features)):
    feature, label = features[i], labels[i]

    # 模型推理
    prediction = forward(feature, weight, bias)
    # 模型验证
    error = mse_loss(prediction, label)
    # 结果输出
    print(f'预测冰淇淋销量：{prediction}')
    print(f'均方差：{error}')
