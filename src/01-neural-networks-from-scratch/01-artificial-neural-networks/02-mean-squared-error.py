import numpy as np

feature = np.array([28.1, 58.0])
label = np.array([165])

weight = np.ones([1, 2]) / 2
bias = np.zeros(1)


def forward(x, w, b):
    return x.dot(w.T) + b


def mse_loss(p, y):
    return ((p - y) ** 2).mean()


prediction = forward(feature, weight, bias)
error = mse_loss(prediction, label)

print(f'prediction: {prediction}')
print(f'error: {error}')
