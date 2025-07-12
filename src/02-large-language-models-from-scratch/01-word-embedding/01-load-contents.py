import re


class DataLoader:

    def __init__(self, filename):
        self.filename = filename

        with open(self.filename, 'r', encoding='utf-8') as f:
            self.text = f.read().lower()

        self.vocabulary = sorted(set(self.split_text(self.text)))
        self.word2index = {word: index for index, word in enumerate(self.vocabulary)}
        self.index2word = {index: word for index, word in enumerate(self.vocabulary)}

    @staticmethod
    def split_text(text):
        words = re.split(r'([,.:;?_!"()\']|\s)', text)
        return [t.strip() for t in words if t.strip()]


dataset = DataLoader('../a-day.txt')

print("Total number of character: ", len(dataset.text))
print("Word to Index: ", dataset.word2index)