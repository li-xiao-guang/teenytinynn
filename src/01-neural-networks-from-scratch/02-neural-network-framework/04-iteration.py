import numpy as np


class DataLoader:

    def __init__(self):
        self.features = [[28.1, 58.0],
                         [22.5, 72.0],
                         [31.4, 45.0],
                         [19.8, 85.0],
                         [27.6, 63]]
        self.labels = [[165],
                       [95],
                       [210],
                       [70],
                       [155]]

    def size(self):
        return len(self.features)

    def feature(self, index):
        return Tensor(self.features[index])

    def label(self, index):
        return Tensor(self.labels[index])

    def feature_size(self):
        return self.feature(0).size()

    def label_size(self):
        return self.label(0).size()


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)
        self.grad = None
        self.gradient_fn = lambda: None
        self.parents = set()

    def gradient(self):
        if self.gradient_fn:
            self.gradient_fn()

        for p in self.parents:
            p.gradient()

    def size(self):
        return len(self.data)


class Linear:

    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size

        self.weight = Tensor(np.ones((out_size, in_size)) / in_size)
        self.bias = Tensor(np.zeros(out_size))

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        p = Tensor(x.data.dot(self.weight.data.T) + self.bias.data)

        def gradient_fn():
            self.weight.grad = p.grad * x.data
            self.bias.grad = np.sum(p.grad, axis=0)

        p.gradient_fn = gradient_fn
        p.parents = {self.weight, self.bias}
        return p

    def parameters(self):
        return [self.weight, self.bias]


class MSELoss:

    def __call__(self, p: Tensor, y: Tensor):
        mse = Tensor(((p.data - y.data) ** 2).mean())

        def gradient_fn():
            p.grad = (p.data - y.data) * 2

        mse.gradient_fn = gradient_fn
        mse.parents = {p}
        return mse


class SGD:

    def __init__(self, parameters):
        self.parameters = parameters

    def backward(self):
        for p in self.parameters:
            p.data -= p.grad


dataset = DataLoader()

model = Linear(dataset.feature_size(), dataset.label_size())
loss = MSELoss()
sgd = SGD(model.parameters())

for i in range(dataset.size()):
    feature, label = dataset.feature(i), dataset.label(i)

    prediction = model(feature)
    error = loss(prediction, label)
    error.gradient()
    sgd.backward()

    print(f'prediction: {prediction.data}')
    print(f'error: {error.data}')
    print(f"weight: {model.weight.data}")
    print(f"bias: {model.bias.data}")
