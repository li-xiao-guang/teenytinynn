import numpy as np

feature = np.array([28.1, 58.0])

weight, bias = np.ones([1, 2]) / 2, np.zeros(1)


def forward(x, w, b):
    return x.dot(w.T) + b


prediction = forward(feature, weight, bias)

print(f'prediction: {prediction}')
