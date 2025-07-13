from abc import ABC, abstractmethod

import numpy as np

np.random.seed(99)


class DataLoader:

    def __init__(self, batch_size):
        self.batch_size = batch_size

        with (np.load('mini-mnist.npz', allow_pickle=True) as f):
            self.x_train, self.y_train = self.normalize(f['x_train'], f['y_train'])
            self.x_test, self.y_test = self.normalize(f['x_test'], f['y_test'])

        self.features = self.x_train
        self.labels = self.y_train

    @staticmethod
    def normalize(x, y):
        inputs = x / 255
        targets = np.zeros((len(y), 10))
        targets[range(len(y)), y] = 1
        return inputs, targets

    def train(self):
        self.features = self.x_train
        self.labels = self.y_train

    def eval(self):
        self.features = self.x_test
        self.labels = self.y_test

    def size(self):
        return len(self.features)

    def feature(self, index):
        return Tensor(self.features[index: index + self.batch_size])

    def label(self, index):
        return Tensor(self.labels[index: index + self.batch_size])

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
        return np.prod(self.data.shape[1:])


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
        p = Tensor(x.data @ self.weight.data.T + self.bias.data)

        def gradient_fn():
            self.weight.grad = p.grad.T @ x.data
            self.bias.grad = np.sum(p.grad, axis=0)
            x.grad = p.grad @ self.weight.data

        p.gradient_fn = gradient_fn
        p.parents = {self.weight, self.bias, x}
        return p

    def parameters(self):
        return [self.weight, self.bias]


class Flatten(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.array(x.data.reshape(x.data.shape[0], -1)))

        def gradient_fn():
            x.grad = p.grad.reshape(x.data.shape)

        p.gradient_fn = gradient_fn
        p.parents = {x}
        return p


class ReLU(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.maximum(0, x.data))

        def gradient_fn():
            x.grad = (p.data > 0) * p.grad

        p.gradient_fn = gradient_fn
        p.parents = {x}
        return p


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


LEARNING_RATE = 0.01
EPOCHES = 10
BATCHES = 2

dataset = DataLoader(BATCHES)

model = Sequential([Flatten(),
                    Linear(dataset.feature_size(), 64),
                    Linear(64, dataset.label_size())])
loss = MSELoss()
sgd = SGD(model.parameters(), LEARNING_RATE)

for epoch in range(EPOCHES):
    for i in range(0, dataset.size(), dataset.batch_size):
        feature = dataset.feature(i)
        label = dataset.label(i)

        prediction = model(feature)
        error = loss(prediction, label)
        error.gradient()
        sgd.backward()

    print(f"epoch: {epoch}")
    print(f'prediction: {prediction.data}')
    print(f'error: {error.data}')
    print(f"hidden weight: {model.layers[1].weight.data}")
    print(f"hidden bias: {model.layers[1].bias.data}")
    print(f"output weight: {model.layers[2].weight.data}")
    print(f"output bias: {model.layers[2].bias.data}")

dataset.eval()

prediction = model(Tensor(dataset.features))
result = (prediction.data.argmax(axis=1) == dataset.labels.argmax(axis=1)).sum()
print(f'Result: {result} of {len(dataset.features)}')
