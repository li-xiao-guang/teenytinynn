import csv
import re
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

np.random.seed(99)


class DataLoader:

    def __init__(self):
        self.reviews = []
        self.sentiments = []
        with open('reviews.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for _, row in enumerate(reader):
                self.reviews.append(row[0])
                self.sentiments.append(row[1])

        split_reviews = []
        for r in self.reviews:
            split_reviews.append(self.clean_text(r.lower()).split())

        counter = Counter([w for r in split_reviews for w in r])
        self.vocabulary = set([w for w, c in counter.items()])
        self.word2index = {w: idx for idx, w in enumerate(self.vocabulary)}
        self.index2word = {idx: w for idx, w in enumerate(self.vocabulary)}
        self.tokens = [[self.word2index[w] for w in r if w in self.word2index] for r in split_reviews]

        self.features = [list(set(idx)) for idx in self.tokens[:-10]]
        self.labels = [0 if idx == "negative" else 1 for idx in self.sentiments[:-10]]

    @staticmethod
    def clean_text(text):
        txt = re.sub(r'<[^>]+>', '', text)
        txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt)
        return txt

    def train(self):
        self.features = [list(set(idx)) for idx in self.tokens[:-10]]
        self.labels = [0 if idx == "negative" else 1 for idx in self.sentiments[:-10]]

    def eval(self):
        self.features = [list(set(idx)) for idx in self.tokens[-10:]]
        self.labels = [0 if idx == "negative" else 1 for idx in self.sentiments[-10:]]

    def size(self):
        return len(self.features)

    def feature(self, index):
        return Tensor(self.features[index: index + 1])

    def label(self, index):
        return Tensor(self.labels[index: index + 1])


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

    def shape(self):
        return self.data.shape

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


class Embedding(Layer):

    def __init__(self, vocabulary_size, embedding_size, axis=1):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.axis = axis

        self.weight = Tensor(np.random.rand(embedding_size, vocabulary_size) / vocabulary_size)

    def forward(self, x: Tensor):
        p = Tensor(np.sum(self.weight.data.T[x.data], axis=self.axis))

        def gradient_fn():
            if self.weight.grad is None:
                self.weight.grad = np.zeros_like(self.weight.data)
            self.weight.grad.T[x.data] += p.grad

        p.gradient_fn = gradient_fn
        p.parents = {self.weight}
        return p

    def parameters(self):
        return [self.weight]


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
                x.grad[idx] = (np.diagflat(itm) - itm @ itm.T) @ p.grad[idx]

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


class BCELoss:

    def __call__(self, p: Tensor, y: Tensor):
        clipped = np.clip(p.data, 1e-7, 1 - 1e-7)
        bce = Tensor(-np.mean(y.data * np.log(clipped)
                              + (1 - y.data) * np.log(1 - clipped)))

        def gradient_fn():
            p.grad = (clipped - y.data) / (clipped * (1 - clipped) * len(p.data))

        bce.gradient_fn = gradient_fn
        bce.parents = {p}
        return bce


class SGD:

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def backward(self):
        for p in self.parameters:
            p.data -= p.grad * self.lr


LEARNING_RATE = 0.1
EPOCHS = 10

dataset = DataLoader()

model = Sequential([Embedding(len(dataset.vocabulary), 64),
                    Tanh(),
                    Linear(64, 16),
                    Tanh(),
                    Linear(16, 1),
                    Sigmoid()])
loss = BCELoss()
sgd = SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    for i in range(dataset.size()):
        feature = dataset.feature(i)
        label = dataset.label(i)

        prediction = model(feature)
        error = loss(prediction, label)
        error.gradient()
        sgd.backward()

    print(f"Epoch: {epoch}")
    print(f'Prediction: {prediction.data}')
    print(f'Error: {error.data}')
    print(f"hidden weight: {model.layers[2].weight.data}")
    print(f"hidden bias: {model.layers[2].bias.data}")
    print(f"output weight: {model.layers[4].weight.data}")
    print(f"output bias: {model.layers[4].bias.data}")

dataset.eval()

result = 0
for i in range(dataset.size()):
    feature = dataset.feature(i)
    label = dataset.label(i)

    prediction = model(feature)
    if np.abs(prediction.data - label.data) < 0.5:
        result += 1
print(f'Result: {result} of {dataset.size()}')
