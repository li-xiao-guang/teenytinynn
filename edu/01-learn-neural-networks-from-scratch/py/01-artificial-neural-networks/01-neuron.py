import numpy as np

# 输入特征
features = np.array([28.1, 58.0])

# 初始化模型参数
weights = np.array([[0.5, 0.5]])
bias = np.array([0])


# 前向传播函数
def forward(x, w, b):
    return x.dot(w.T) + b


# 模型推理
prediction = forward(features, weights, bias)
# 输出结果
print(f'预测的冰淇淋销量为: {prediction}')
