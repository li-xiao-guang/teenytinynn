import numpy as np

features = np.array([[28.1, 58.0],
                     [22.5, 72.0],
                     [31.4, 45.0],
                     [19.8, 85.0],
                     [27.6, 63]])
labels = np.array([[165],
                   [95],
                   [210],
                   [70],
                   [155]])

weight, bias = np.ones([1, 2]) / 2, np.zeros(1)


def forward(x, w, b):
    return x.dot(w.T) + b


def mse_loss(p, y):
    return ((p - y) ** 2).mean()


def gradient(p, y):
    return (p - y) * 2


def backward(x, d, w, b, lr):
    return w - d * x * lr, b - np.sum(d) * lr


LEARNING_RATE = 0.00001
EPOCHES = 1000

for epoch in range(EPOCHES):
    for i in range(len(features)):
        feature, label = features[i], labels[i]

        prediction = forward(feature, weight, bias)
        error = mse_loss(prediction, label)
        delta = gradient(prediction, label)
        weight, bias = backward(feature, delta, weight, bias, LEARNING_RATE)

    print(f"epoch: {epoch}")
    print(f'prediction: {prediction}')
    print(f'error: {error}')
    print(f"weight: {weight}")
    print(f"bias: {bias}")
