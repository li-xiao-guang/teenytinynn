import numpy as np


class DataLoader:

    def __init__(self):
        self.feature = Tensor([28.1, 58.0])

    def feature_size(self):
        return self.feature.size()


class Tensor:

    def __init__(self, data):
        self.data = np.array(data)

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
        return Tensor(x.data.dot(self.weight.data.T) + self.bias.data)


dataset = DataLoader()

model = Linear(dataset.feature_size(), 1)

prediction = model(dataset.feature)

print(f'prediction: {prediction.data}')
