import numpy as np


class DataLoader:

    def __init__(self):
        self.feature = Tensor([28.1, 58.0])
        self.label = Tensor([165])


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


class MSELoss:

    def __call__(self, p: Tensor, y: Tensor):
        return Tensor(((p.data - y.data) ** 2).mean())


dataset = DataLoader()

model = Linear(dataset.feature.size(), dataset.label.size())
loss = MSELoss()

prediction = model(dataset.feature)
error = loss(prediction, dataset.label)

print(f'prediction: {prediction.data}')
print(f'error: {error.data}')
