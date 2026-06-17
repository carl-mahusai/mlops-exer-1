import torch
from torch.utils.data import Dataset
from collections import Counter
import re

from spam_checker.data.util import build_vocab_util

class SMSDataset(Dataset):
    # def __init__(self, data, vocab, max_length=50):
    #     self.texts = data['text']
    #     self.labels = data['label']
    def __init__(self, texts, labels, vocab, max_length=50, max_vocab_size=2):
        self.texts = texts
        self.labels = labels
        self.vocab = None
        self.max_length = max_length
        self.max_vocab_size = max_vocab_size

        if (not vocab):
            self.vocab = self.build_vocab(texts=texts)
        else:
            self.vocab = vocab
        
        print("calling smsdataset")

    def build_vocab(self, texts):
        return build_vocab_util(
            texts=texts,
            max_vocab_size=self.max_vocab_size
        )

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