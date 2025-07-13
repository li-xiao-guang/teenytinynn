import re
from abc import ABC, abstractmethod

import numpy as np

np.random.seed(99)


class DataLoader:

    def __init__(self, filename, batch_size, stride):
        self.filename = filename
        self.batch_size = batch_size
        self.stride = stride

        with open(self.filename, 'r', encoding='utf-8') as f:
            self.text = f.read().lower()

        self.vocabulary = sorted(set(self.split_text(self.text)))
        self.vocabulary.extend(["<|eos|>", "<|unk|>"])
        self.word2index = {word: index for index, word in enumerate(self.vocabulary)}
        self.index2word = {index: word for index, word in enumerate(self.vocabulary)}
        self.tokens = self.encode(self.text)

        self.features = []
        self.labels = []

        for i in range(0, len(self.tokens) - self.batch_size, self.stride):
            self.features.append(self.tokens[i: i + self.batch_size])
            self.labels.append(self.tokens[i + 1: i + self.batch_size + 1])

    @staticmethod
    def split_text(text):
        words = re.split(r'([,.:;?_!"()\']|\s)', text.lower())
        return [t.strip() for t in words if t.strip()]

    def encode(self, text):
        words = self.split_text(text)
        words = [word if word in self.word2index else "<|unk|>" for word in words]
        return [self.word2index[word] for word in words]

    def decode(self, tokens):
        text = " ".join([self.index2word[index] for index in tokens])
        return re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)

    def __len__(self):  # 3
        return len(self.features)

    def __getitem__(self, index):  # 4
        return self.features[index], self.labels[index]


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

    def __add__(self, other):
        p = Tensor(self.data + other.data)

        def gradient_fn():
            self.grad = p.grad
            other.grad = p.grad

        p.gradient_fn = gradient_fn
        p.parents = {self, other}
        return p

    def __sub__(self, other):
        p = Tensor(self.data - other.data)

        def gradient_fn():
            self.grad = p.grad
            other.grad = -p.grad

        p.gradient_fn = gradient_fn
        p.parents = {self, other}
        return p

    def __mul__(self, other):
        p = Tensor(self.data * other.data)

        def gradient_fn():
            self.grad = p.grad * other.data
            other.grad = p.grad * self.data

        p.gradient_fn = gradient_fn
        p.parents = {self, other}
        return p

    def __truediv__(self, other):
        p = Tensor(self.data / other.data)

        def gradient_fn():
            self.grad = p.grad / other.data
            other.grad = -p.grad * self.data / (other.data ** 2)

        p.gradient_fn = gradient_fn
        p.parents = {self, other}
        return p

    def __matmul__(self, other):
        p = Tensor(np.matmul(self.data, other.data))

        def gradient_fn():
            self.grad = np.matmul(p.grad, other.data.T)
            other.grad = np.matmul(self.data.T, p.grad)

        p.gradient_fn = gradient_fn
        p.parents = {self, other}
        return p

    def transpose(self, axes=None):
        p = Tensor(np.transpose(self.data, axes))

        def gradient_fn():
            if axes is None:
                self.grad = np.transpose(p.grad)
            else:
                idx = np.argsort(axes)
                self.grad = np.transpose(p.grad, idx)

        p.gradient_fn = gradient_fn
        p.parents = {self}
        return p

    @property
    def T(self):
        return self.transpose()

    def concat(self, other, axis):
        p = Tensor(np.concatenate([self.data, other.data], axis=axis))

        def gradient_fn():
            self.grad, other.grad = np.split(p.grad, [self.data.shape[axis]], axis=axis)

        p.gradient_fn = gradient_fn
        p.parents = {self, other}
        return p


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


class Linear(Layer):

    def __init__(self, in_size, out_size, bias_enabled=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.bias_enabled = bias_enabled

        self.weight = Tensor(np.random.rand(out_size, in_size) / in_size)
        self.bias = Tensor(np.zeros(out_size))

    def __call__(self, x: Tensor):
        return self.forward(x)

    def forward(self, x: Tensor):
        p = Tensor(x.data @ self.weight.data.T + self.bias.data)

        def gradient_fn():
            self.weight.grad = p.grad.T @ x.data
            if self.bias_enabled:
                self.bias.grad = np.sum(p.grad, axis=0)
            x.grad = p.grad @ self.weight.data

        p.gradient_fn = gradient_fn
        p.parents = {self.weight, x}
        if self.bias_enabled:
            p.parents.add(self.bias)
        return p

    def parameters(self):
        parameters = [self.weight]
        if self.bias_enabled:
            parameters.append(self.bias)
        return parameters


class Embedding(Layer):

    def __init__(self, vocabulary_size, embedding_size, axis=1):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.axis = axis

        self.weight = Tensor(np.random.rand(embedding_size, vocabulary_size) / vocabulary_size)

    def forward(self, x: Tensor):
        p = Tensor(self.weight.data.T[x.data])

        def gradient_fn():
            if self.weight.grad is None:
                self.weight.grad = np.zeros_like(self.weight.data)
            self.weight.grad.T[x.data] += p.grad

        p.gradient_fn = gradient_fn
        p.parents = {self.weight}
        return p

    def parameters(self):
        return [self.weight]


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


class Triu(Layer):

    def __init__(self, size, value):
        super().__init__()
        self.size = size
        self.value = value

        self.mask = np.triu(np.ones((self.size, self.size))).T

    def forward(self, x: Tensor):
        if not self.training:
            return x

        p = Tensor(x.data)
        p.data[self.mask == 0] = self.value

        def gradient_fn():
            x.grad = p.grad * self.mask

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


CONTEXT_SIZE = 6

dataset = DataLoader('../a-day.txt', CONTEXT_SIZE, 1)

token_embedding = Embedding(len(dataset.vocabulary), 4, 0)
positional_embedding = Embedding(CONTEXT_SIZE, 4, 0)

feature, label = dataset[0]

feature_token = token_embedding(Tensor(feature))
feature_position = positional_embedding(Tensor(range(CONTEXT_SIZE)))
feature_embedding = feature_token + feature_position
print("Embedding: ", feature_embedding.data)

query = Linear(feature_embedding.size(), 2, bias_enabled=False)
key = Linear(feature_embedding.size(), 2, bias_enabled=False)
value = Linear(feature_embedding.size(), 2, bias_enabled=False)

query_feature = query(feature_embedding)
key_feature = key(feature_embedding)
value_feature = value(feature_embedding)

attention_scores = query_feature @ key_feature.T
print("Attention scores: ", attention_scores.data)

triu = Triu(CONTEXT_SIZE, -np.inf)
attention_scores = triu(attention_scores)

softmax = Softmax()
attention_weights = softmax(attention_scores)
dropout = Dropout()
attention_weights = dropout(attention_weights)
print("Attention weights: ", attention_weights.data)
print("Attention weight sum: ", attention_weights.data.sum(axis=1))

context_vectors = attention_weights @ value_feature
print("Context vectors: ", context_vectors.data)
