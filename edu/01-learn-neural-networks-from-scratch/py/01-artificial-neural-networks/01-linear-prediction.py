import numpy as np

# 输入数据
feature = np.array([28.1, 58.0])

# 模型参数（权重和偏置）
weight, bias = np.array([1, 1]), np.array([0])

# 神经元逻辑
forward = lambda x: x.dot(weight.T) + bias

# 模型推理
prediction = forward(feature)

# 结果输出
print(f'预测冰淇淋销量：{prediction}')
