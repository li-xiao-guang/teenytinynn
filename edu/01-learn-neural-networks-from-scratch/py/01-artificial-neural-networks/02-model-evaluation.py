import numpy as np

# 输入数据
feature = np.array([28.1, 58.0])
# 实际结果
label = np.array([165])
# 模型参数（权重和偏置）
weight, bias = np.array([1, 1]), np.array([0])


# 神经元逻辑（线性回归（多元一次）函数）
def forward(x, w, b):
    return x.dot(w.T) + b


# 损失函数（平均平方差）
def mse_loss(p, y):
    return ((p - y) ** 2).mean()


# 模型推理
prediction = forward(feature, weight, bias)
# 模型验证
error = mse_loss(prediction, label)
# 结果输出
print(f'预测冰淇淋销量：{prediction}')
print(f'平均平方差：{error}')
