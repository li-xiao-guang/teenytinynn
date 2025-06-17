import numpy as np

# 输入数据
feature = np.array([28.1, 58.0])
# 模型参数（权重和偏置）
weight, bias = np.array([[1, 1]]) / 2, np.array([0])


# 神经元逻辑（线性回归（多元一次）函数）
def forward(x, w, b):
    return x.dot(w.T) + b


# 模型推理
prediction = forward(feature, weight, bias)
# 结果输出
print(f'预测冰淇淋销量：{prediction}')
