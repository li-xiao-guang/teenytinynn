from abc import abstractmethod, ABC

import numpy as np

np.random.seed(99)


class DataLoader:

    def __init__(self, batch_size):
        self.batch_size = batch_size

        with (np.load('mini-mnist.npz', allow_pickle=True) as f):
            self.features, self.labels = self.normalize(f['x_train'], f['y_train'])
            self.observations, self.actuals = self.normalize(f['x_test'], f['y_test'])

    @staticmethod
    def normalize(x, y):
        inputs = x / 255
        targets = np.zeros((len(y), 10))
        targets[range(len(y)), y] = 1
        return inputs, targets

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
        return np.prod(self.data.shape[1:])


class Layer(ABC):

    def __init__(self):
        self.training = True

    def __call__(self, x: Tensor):
        return self.forward(x)

    @abstractmethod
    def forward(self, x: Tensor):
        pass

    def parameters(self):
        return []

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class Sequential(Layer):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def train(self):
        for l in self.layers:
            l.train()

    def eval(self):
        for l in self.layers:
            l.eval()


class Linear(Layer):

    def __init__(self, in_size, out_size):
        super().__init__()
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


class Flatten(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.array(x.data.reshape(x.data.shape[0], -1)))

        def gradient_fn():
            x.grad = p.grad.reshape(x.data.shape)

        p.gradient_fn = gradient_fn
        p.parents = {x}
        return p


class Dropout(Layer):

    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x: Tensor):
        if not self.training:
            return x

        mask = np.random.random(x.data.shape) > self.dropout_rate
        p = Tensor(x.data * mask)

        def gradient_fn():
            x.grad = p.grad * mask

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


class Tanh(Layer):

    def forward(self, x: Tensor):
        p = Tensor(np.tanh(x.data))

        def gradient_fn():
            x.grad = p.grad * (1 - p.data ** 2)

        p.gradient_fn = gradient_fn
        p.parents = {x}
        return p


class Sigmoid(Layer):

    def __init__(self, clip_range=(-100, 100)):
        super().__init__()
        self.clip_range = clip_range

    def forward(self, x: Tensor):
        z = np.clip(x.data, self.clip_range[0], self.clip_range[1])
        p = Tensor(1 / (1 + np.exp(-z)))

        def gradient_fn():
            x.grad = p.grad * p.data * (1 - p.data)

        p.gradient_fn = gradient_fn
        p.parents = {x}
        return p


class Softmax(Layer):

    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        exp = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        p = Tensor(exp / np.sum(exp, axis=self.axis, keepdims=True))

        def gradient_fn():
            x.grad = np.zeros_like(x.data)
            for idx in range(x.data.shape[0]):
                itm = p.data[idx].reshape(-1, 1)
                x.grad[idx] = (np.diagflat(itm) - itm.dot(itm.T)).dot(p.grad[idx])

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
                    Dropout(),
                    Linear(dataset.feature(0).size(), 64),
                    Tanh(),
                    Linear(64, dataset.label(0).size()),
                    Softmax()])
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
    print(f"hidden weight: {model.layers[2].weight.data}")
    print(f"hidden bias: {model.layers[2].bias.data}")
    print(f"output weight: {model.layers[4].weight.data}")
    print(f"output bias: {model.layers[4].bias.data}")

model.eval()
prediction = model(Tensor(dataset.observations))
result = (prediction.data.argmax(axis=1) == dataset.actuals.argmax(axis=1)).sum()
print(f'Result: {result} of {len(dataset.observations)}')
