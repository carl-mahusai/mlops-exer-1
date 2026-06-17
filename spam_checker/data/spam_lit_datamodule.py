from collections import Counter
import torch
import lightning as L
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from spam_checker.data.spam_dataset import SMSDataset

from spam_checker.data.util import build_vocab_util


class SMSDataModule(L.LightningDataModule):
    def __init__(self, dataframe, batch_size=8, max_vocab_size=10000, max_length=50, num_workers = 0, persistent_workers = False, vocab = None):
        super().__init__()

        self.df = dataframe.copy()

        self.batch_size = batch_size
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.num_workers = num_workers

        self.vocab = vocab

        self.persistent_workers = persistent_workers

        # self.gpus = int(torch.cuda.is_available())

        # self.on_gpu = False

        # if (isinstance(self.gpus, (str, int)) and self.gpus > 0):
        #     self.on_gpu = True


    # def build_vocab(self, texts):
    #     counter = Counter()

    #     for text in texts:
    #         counter.update(
    #             str(text).lower().split()
    #         )

    #     vocab = {
    #         "<PAD>": 0,
    #         "<UNK>": 1,
    #     }

    #     for token, _ in counter.most_common(
    #         self.max_vocab_size - 2
    #     ):
    #         vocab[token] = len(vocab)

    #     return vocab

    def build_vocab(self, texts):
        return build_vocab_util(
            texts=texts,
            max_vocab_size=self.max_vocab_size
        )

    def setup(self, stage=None):
        df = self.df.copy()

        df["label"] = (
            df["label"]
            .replace({"ham": 0, "spam": 1})
            .astype(int)
        )

        train_df, temp_df = train_test_split(
            df,
            test_size=0.20,
            stratify=df["label"],
            random_state=42,
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            stratify=temp_df["label"],
            random_state=42,
        )

        if (not self.vocab):
            self.vocab = self.build_vocab(
                train_df["message"]
            )

        self.train_dataset = SMSDataset(
            train_df["message"].tolist(),
            train_df["label"].tolist(),
            self.vocab,
            self.max_length,
            self.max_vocab_size
        )

        self.val_dataset = SMSDataset(
            val_df["message"].tolist(),
            val_df["label"].tolist(),
            self.vocab,
            self.max_length,
            self.max_vocab_size
        )

        self.test_dataset = SMSDataset(
            test_df["message"].tolist(),
            test_df["label"].tolist(),
            self.vocab,
            self.max_length,
            self.max_vocab_size
        )

    def train_dataloader(self):
        print("running train dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        print("running val dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        print("running test dataloader")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.persistent_workers
        )