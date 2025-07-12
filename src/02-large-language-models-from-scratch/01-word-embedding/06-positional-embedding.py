import re
from abc import ABC, abstractmethod

import numpy as np


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

    def __mul__(self, other):
        p = Tensor(self.data * other.data)

        def gradient_fn():
            self.grad = p.grad * other.data
            other.grad = p.grad * self.data

        p.gradient_fn = gradient_fn
        p.parents = {self, other}
        return p

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


BATCHES = 4
SEQUENCES = 10

dataset = DataLoader('../a-day.txt', BATCHES, 1)

token_embedding = Embedding(len(dataset.vocabulary), 64, 0)
print(token_embedding.weight.shape())

positional_embedding = Embedding(BATCHES, 64, 0)
print(positional_embedding.weight.shape())

feature, label = dataset[:SEQUENCES]

feature_token = token_embedding(Tensor(feature))
print(feature_token.shape())

feature_position = positional_embedding(Tensor(range(BATCHES)))
print(feature_position.shape())

feature_embedding = feature_token + feature_position
print(feature_embedding.shape())




