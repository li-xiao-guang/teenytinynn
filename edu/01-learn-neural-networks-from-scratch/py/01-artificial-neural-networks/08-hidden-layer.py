import numpy as np

np.random.seed(99)


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


def gradient_backward(d, w):
    return d.dot(w)


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
h_weight = np.random.rand(4, features.shape[-1]) / features.shape[-1]
h_bias = np.zeros(4)

o_weight = np.random.rand(labels.shape[-1], 4) / 4
o_bias = np.zeros(labels.shape[-1])

# epoch
for epoch in range(EPOCHES):
    # iteration
    for i in range(0, len(features), BATCHES):
        feature = features[i: i + BATCHES]
        label = labels[i: i + BATCHES]

        # prediction
        hidden = forward(feature, h_weight, h_bias)
        prediction = forward(hidden, o_weight, o_bias)

        # evaluation
        error = mse_loss(prediction, label)

        # gradient computation
        o_delta = gradient(prediction, label)
        h_delta = gradient_backward(o_delta, o_weight)

        # parameter updates
        o_weight, o_bias = backward(hidden, o_delta, o_weight, o_bias, LEARNING_RATE)
        h_weight, h_bias = backward(feature, h_delta, h_weight, h_bias, LEARNING_RATE)

    print(f"Epoch: {epoch}")
    print(f'Prediction: {prediction}')
    print(f'Error: {error}')
    print(f"Hidden weight: {h_weight}")
    print(f"Hidden bias: {h_bias}")
    print(f"Output weight: {o_weight}")
    print(f"Output bias: {o_bias}")

