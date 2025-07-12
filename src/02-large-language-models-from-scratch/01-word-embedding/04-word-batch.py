import re


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


dataset = DataLoader('../a-day.txt', 4, 1)

for i in range(len(dataset)):
    print(dataset[i])
