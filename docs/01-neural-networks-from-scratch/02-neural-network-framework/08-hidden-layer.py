from abc import abstractmethod, ABC

import numpy as np

np.random.seed(99)


class DataLoader:

    def __init__(self, batch_size):
        self.batch_size = batch_size

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
        return Tensor(self.features[index: index + self.batch_size])

    def label(self, index):
        return Tensor(self.labels[index: index + self.batch_size])


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
        return self.data.shape[-1]


class Layer(ABC):

    def __call__(self, x: Tensor):
        return self.forward(x)

    @abstractmethod
    def forward(self, x: Tensor):
        pass

    def parameters(self):
        return []


class Sequential(Layer):

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x: Tensor):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


class Linear(Layer):

    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size

        self.weight = Tensor(np.random.rand(out_size, in_size) / in_size)
        self.bias = Tensor(np.zeros(out_size))

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        p = Tensor(x.data.dot(self.weight.data.T) + self.bias.data)

        def gradient_fn():
            self.weight.grad = p.grad.T.dot(x.data)
            self.bias.grad = np.sum(p.grad, axis=0)
            x.grad = p.grad.dot(self.weight.data)

        p.gradient_fn = gradient_fn
        p.parents = {self.weight, self.bias, x}
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

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def backward(self):
        for p in self.parameters:
            p.data -= p.grad * self.lr


LEARNING_RATE = 0.00001
EPOCHES = 1000
BATCHES = 2

dataset = DataLoader(BATCHES)

model = Sequential([Linear(dataset.feature(0).size(), 4),
                    Linear(4, dataset.label(0).size())])
loss = MSELoss()
sgd = SGD(model.parameters(), LEARNING_RATE)

for epoch in range(EPOCHES):
    for i in range(0, dataset.size(), dataset.batch_size):
        feature, label = dataset.feature(i), dataset.label(i)

        prediction = model(feature)
        error = loss(prediction, label)
        error.gradient()
        sgd.backward()

    print(f"epoch: {epoch}")
    print(f'prediction: {prediction.data}')
    print(f'error: {error.data}')
    print(f"hidden weight: {model.layers[0].weight.data}")
    print(f"hidden bias: {model.layers[0].bias.data}")
    print(f"output weight: {model.layers[1].weight.data}")
    print(f"output bias: {model.layers[1].bias.data}")
