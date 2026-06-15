import torch
from torch.utils.data import Dataset
from collections import Counter
import re

class SMSDataset(Dataset):
    # def __init__(self, data, vocab, max_length=50):
    #     self.texts = data['text']
    #     self.labels = data['label']
    def __init__(self, texts, labels, vocab, max_length=50):
        self.texts = texts
        self.labels = labels
        self.vocab = None
        self.max_length = max_length

        if (not vocab):
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
        
        print("calling smsdataset")

    def build_vocab(self, texts):
        counter = Counter()

        for text in texts:
            counter.update(
                str(text).lower().split()
            )

        vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
        }

        for token, _ in counter.most_common(
            self.max_vocab_size - 2
        ):
            vocab[token] = len(vocab)

        return vocab

    def encode(self, text):
        tokens = str(text).lower().split()

        ids = [
            self.vocab.get(token, self.vocab["<UNK>"])
            for token in tokens
        ]

        ids = ids[: self.max_length]

        if len(ids) < self.max_length:
            ids.extend(
                [self.vocab["<PAD>"]]
                * (self.max_length - len(ids))
            )

        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = self.encode(self.texts[idx])

        y = torch.tensor(
            self.labels[idx],
            dtype=torch.float32,
        )

        return x, y