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
        self.train()

    @staticmethod
    def split_text(text):
        words = re.split(r'([,.:;?_!"()\']|\s)', text.lower())
        return [t.strip() for t in words if t.strip()]

    def train(self):
        self.features.clear()
        self.labels.clear()
        for i in range(0, len(self.tokens) * 9 // 10 - self.batch_size, self.stride):
            self.features.append(self.tokens[i: i + self.batch_size])
            self.labels.append(self.tokens[i + 1: i + self.batch_size + 1])

    def eval(self):
        self.features.clear()
        self.labels.clear()
        for i in range(len(self.tokens) * 9 // 10 - self.batch_size + 1, len(self.tokens) - self.batch_size,
                       self.stride):
            self.features.append(self.tokens[i: i + self.batch_size])
            self.labels.append(self.tokens[i + 1: i + self.batch_size + 1])

    def __len__(self):  # 3
        return len(self.features)

    def __getitem__(self, index):  # 4
        return self.features[index], self.labels[index]

    def encode(self, text):
        words = self.split_text(text)
        words = [word if word in self.word2index else "<|unk|>" for word in words]
        return [self.word2index[word] for word in words]

    def decode(self, tokens):
        text = " ".join([self.index2word[index] for index in tokens])
        return re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)

    def embedding(self, index):
        ebd = np.zeros(len(self.vocabulary))
        ebd[index] = 1
        return ebd


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
            self.grad = np.matmul(p.grad, other.data.swapaxes(-1, -2))
            other.grad = np.matmul(self.data.swapaxes(-1, -2), p.grad)

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

    def reshape(self, shape):
        p = Tensor(np.reshape(self.data, shape))

        def gradient_fn():
            self.grad = np.reshape(p.grad, self.data.shape)

        p.gradient_fn = gradient_fn
        p.parents = {self}
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

    def __init__(self, in_size, out_size, bias_enabled=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.bias_enabled = bias_enabled

        self.weight = Tensor(np.random.rand(out_size, in_size) / in_size)
        self.bias = Tensor(np.zeros(out_size))

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


class Normalize(Layer):

    def __init__(self, size, eps=0.00001):
        super().__init__()
        self.size = size
        self.eps = eps

        self.scale = Tensor(np.ones(self.size))
        self.shift = Tensor(np.zeros(self.size))

    def forward(self, x: Tensor):
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True, ddof=0)
        norm = (x.data - mean) / np.sqrt(var + self.eps)
        p = Tensor(self.scale.data * norm + self.shift.data)

        def gradient_fn():
            axis = tuple(range(p.grad.ndim - 1))
            self.scale.grad = np.sum(p.grad * norm, axis=axis)
            self.shift.grad = np.sum(p.grad, axis=axis)

            grad = p.grad * self.scale.data
            grad_mean = np.mean(grad, axis=-1, keepdims=True)
            norm_mean = np.mean(grad * norm, axis=-1, keepdims=True)
            x.grad = (grad - grad_mean - norm * norm_mean) / np.sqrt(var + self.eps)

        p.gradient_fn = gradient_fn
        p.parents = {self.scale, self.shift, x}
        return p

    def parameters(self):
        return [self.scale, self.shift]


class Triu(Layer):

    def __init__(self, value=-np.inf):
        super().__init__()
        self.value = value

    def forward(self, x: Tensor):
        if not self.training:
            return x

        axes = list(range(x.data.ndim))
        axes[-2], axes[-1] = axes[-1], axes[-2]
        mask = np.triu(np.ones(x.data.shape)).transpose(axes)
        p = Tensor(x.data)
        p.data[mask == 0] = self.value

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


class GeLU(Layer):

    def forward(self, x: Tensor):
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        tanh_input = sqrt_2_over_pi * (x.data + 0.044715 * x.data ** 3)
        gelu_output = 0.5 * x.data * (1 + np.tanh(tanh_input))

        p = Tensor(gelu_output)

        def gradient_fn():
            tanh_val = np.tanh(tanh_input)
            sech2_val = 1 - tanh_val ** 2
            dtanh_input_dx = sqrt_2_over_pi * (1 + 3 * 0.044715 * x.data ** 2)
            gelu_grad = 0.5 * (1 + tanh_val) + 0.5 * x.data * sech2_val * dtanh_input_dx
            x.grad = gelu_grad * p.grad

        p.gradient_fn = gradient_fn
        p.parents = {x}
        return p


class Softmax(Layer):

    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor):
        exp = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        p = Tensor(exp / np.sum(exp, axis=self.axis, keepdims=True))

        def gradient_fn():
            x.grad = np.zeros_like(x.data)

            shape = x.data.shape
            axis = self.axis if self.axis >= 0 else len(shape) + self.axis
            shapes = list(shape)
            shapes.pop(axis)

            for idx in np.ndindex(tuple(shapes)):
                indices = list(idx)
                indices.insert(axis, slice(None))
                indices = tuple(indices)

                probs = p.data[indices]
                grad = p.grad[indices]

                probs_col = probs.reshape(-1, 1)
                jacobian = np.diagflat(probs) - probs_col @ probs_col.T
                x.grad[indices] = jacobian @ grad

        p.gradient_fn = gradient_fn
        p.parents = {x}
        return p


class CELoss:

    def __call__(self, p: Tensor, y: Tensor):
        exp = np.exp(p.data - np.max(p.data, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=-1, keepdims=True)

        log = np.log(softmax + 1e-10)
        ce = Tensor(0 - np.sum(y.data * log) / len(p.data))

        def gradient_fn():
            p.grad = (softmax - y.data) / len(p.data)

        ce.gradient_fn = gradient_fn
        ce.parents = {p}
        return ce


class Adam:

    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        self.m = [None for _ in range(len(params))]
        self.v = [None for _ in range(len(params))]
        self.t = 0

    def backward(self):
        self.t += 1
        for idx, p in enumerate(self.parameters):
            if p is not None and p.grad is not None:
                grad = p.grad.reshape(p.data.shape)

                if self.m[idx] is None:
                    self.m[idx] = np.zeros_like(p.data)
                    self.v[idx] = np.zeros_like(p.data)

                self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
                self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)
                m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
                p.data -= m_hat / (np.sqrt(v_hat) + self.eps) * self.lr


class GPT(Layer):

    def __init__(self, vocabulary_size, context_size, embedding_size, heads):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.heads = heads

        self.embedding = GPTEmbedding(self.vocabulary_size, self.context_size, self.embedding_size)
        self.attention = GPTAttention(self.context_size, self.embedding_size, self.heads)
        self.feed_forward = GPTFeedForward(self.embedding_size)
        self.output = GPTOutput(self.vocabulary_size, self.embedding_size)

        self.layers = [self.embedding,
                       self.attention,
                       self.feed_forward,
                       self.output]

    def forward(self, x: Tensor):
        x = self.embedding(x)
        x = self.attention(x)
        x = self.feed_forward(x)
        return self.output(x)


class GPTEmbedding(Layer):

    def __init__(self, vocabulary_size, context_size, embedding_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.context_size = context_size
        self.embedding_size = embedding_size

        self.token_embedding = Embedding(self.vocabulary_size, self.embedding_size)
        self.positional_embedding = Embedding(self.context_size, self.embedding_size)

        self.layers = [self.token_embedding,
                       self.positional_embedding]

    def forward(self, x: Tensor):
        token = self.token_embedding(x)
        position = self.positional_embedding(Tensor(range(self.context_size)))
        return token + position


class GPTAttention(Layer):

    def __init__(self, context_size, embedding_size, heads=1):
        super().__init__()
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.heads = heads

        self.normalize = Normalize(self.embedding_size)
        self.query = Linear(self.embedding_size, self.embedding_size * self.heads)
        self.key = Linear(self.embedding_size, self.embedding_size * self.heads)
        self.value = Linear(self.embedding_size, self.embedding_size * self.heads)
        self.triu = Triu(self.context_size)
        self.softmax = Softmax()
        self.merge = Linear(self.heads * self.embedding_size, self.embedding_size)

        self.layers = [self.normalize,
                       self.query,
                       self.key,
                       self.value,
                       self.triu,
                       self.softmax,
                       self.merge]

    def forward(self, x: Tensor):
        x = self.normalize(x)

        query = self.query(x).reshape((-1, self.heads, self.embedding_size))
        key = self.key(x).reshape((-1, self.heads, self.embedding_size))
        value = self.value(x).reshape((-1, self.heads, self.embedding_size))

        scores = self.triu(query @ key.transpose((0, 2, 1)))
        weights = self.softmax(scores)
        vectors = self.merge((weights @ value).reshape((-1, self.heads * self.embedding_size)))
        return vectors


class GPTFeedForward(Layer):

    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.normalize = Normalize(self.embedding_size)
        self.feed_forward = Sequential([Linear(self.embedding_size, self.embedding_size * 4),
                                        GeLU(),
                                        Linear(self.embedding_size * 4, self.embedding_size)])

        self.layers = [self.normalize,
                       self.feed_forward]

    def forward(self, x: Tensor):
        x = self.normalize(x)
        return self.feed_forward(x)


class GPTOutput(Layer):

    def __init__(self, vocabulary_size, embedding_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

        self.normalize = Normalize(self.embedding_size)
        self.output = Linear(self.embedding_size, self.vocabulary_size)

        self.layers = [self.normalize,
                       self.output]

    def forward(self, x: Tensor):
        x = self.normalize(x)
        return self.output(x)


CONTEXT_SIZE = 4
EMBEDDING_SIZE = 4
HEADS = 2
LEARNING_RATE = 0.0000001
EPOCHS = 10

dataset = DataLoader('../a-day.txt', CONTEXT_SIZE, 1)

model = GPT(len(dataset.vocabulary), CONTEXT_SIZE, EMBEDDING_SIZE, HEADS)

loss = CELoss()
sgd = Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    for i in range(len(dataset)):
        feature, label = dataset[i]

        prediction = model(Tensor(feature))
        error = loss(prediction, Tensor(dataset.embedding(label)))
        error.gradient()
        sgd.backward()

    print(f"Epoch: {epoch}")
    print(f'Prediction: {prediction.data}')
    print(f'Error: {error.data}')

dataset.eval()

for i in range(len(dataset)):
    feature, label = dataset[i]

    prediction = model(Tensor(feature))

    print("Feature: ", dataset.decode(feature))
    print("Label: ", dataset.decode(label))
    tokens = []
    for i in range(len(label)):
        tokens.append(prediction.data[i].argmax())
    print("Prediction: ", dataset.decode(tokens))
