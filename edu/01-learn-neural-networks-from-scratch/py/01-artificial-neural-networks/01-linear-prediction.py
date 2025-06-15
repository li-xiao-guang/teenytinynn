import numpy as np

# dataset
feature = np.array([28.1, 58.0])

# model
weight, bias = np.array([1.0, 1.0]), np.array([0.0])

# neuron
forward = lambda x: x.dot(weight.T) + bias

# predict
prediction = forward(feature)

print(f'Prediction: {prediction}')
