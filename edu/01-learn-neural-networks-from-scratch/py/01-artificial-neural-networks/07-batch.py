import numpy as np


# neuron definition
def forward(x, w, b):
    return x.dot(w.T) + b


def backward(x, d, w, b, lr):
    return w - d.T.dot(x) * lr, b - np.sum(d, axis=0) * lr


# loss function
def mse_loss(p, y):
    return ((p - y) ** 2).mean()


# gradient descent
def gradient(p, y):
    return (p - y) * 2 / len(y)


# dataset
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

# hyperparameters
LEARNING_RATE = 0.00001
EPOCHES = 1000
BATCHES = 2

# model
weight = np.ones((labels.shape[-1], features.shape[-1]))
bias = np.zeros(labels.shape[-1])

# epoch
for epoch in range(EPOCHES):
    # iteration
    for i in range(0, len(features), BATCHES):
        feature = features[i: i + BATCHES]
        label = labels[i: i + BATCHES]

        # prediction
        prediction = forward(feature, weight, bias)

        # evaluation
        error = mse_loss(prediction, label)

        # gradient computation
        delta = gradient(prediction, label)

        # parameter updates
        weight, bias = backward(feature, delta, weight, bias, LEARNING_RATE)

    print(f"Epoch: {epoch}")
    print(f'Prediction: {prediction}')
    print(f'Error: {error}')
    print(f"Weight: {weight}")
    print(f"Bias: {bias}")
