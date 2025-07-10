import numpy as np

feature = np.array([28.1, 58.0])
label = np.array([165])

weight, bias = np.ones([1, 2]) / 2, np.zeros(1)


def forward(x, w, b):
    return x.dot(w.T) + b


def mse_loss(p, y):
    return ((p - y) ** 2).mean()


def gradient(p, y):
    return (p - y) * 2


def backward(x, d, w, b):
    return w - d * x, b - np.sum(d)


prediction = forward(feature, weight, bias)
error = mse_loss(prediction, label)
delta = gradient(prediction, label)
weight, bias = backward(feature, delta, weight, bias)

print(f'prediction: {prediction}')
print(f'error: {error}')
print(f"weight: {weight}")
print(f"bias: {bias}")
