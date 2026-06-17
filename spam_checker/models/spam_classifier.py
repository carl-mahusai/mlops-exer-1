import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpamClassifier(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        # embedding_dim=128,
        # hidden_dim=64,
        embedding_dim=64,
        hidden_dim=64,
        lr=1e-3,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.fc1 = nn.Linear(
            embedding_dim,
            hidden_dim,
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(self, x):
        x = self.embedding(x)

        # Mean pooling
        x = x.mean(dim=1)

        x = F.relu(self.fc1(x))

        logits = self.fc2(x)

        return logits.squeeze(1)

    def shared_step(self, batch):
        x, y = batch

        logits = self(x)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
        )

        preds = (
            torch.sigmoid(logits) > 0.5
        ).float()

        acc = (preds == y).float().mean()

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_acc", acc, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True
        )

        self.log(
            "val_acc",
            acc,
            prog_bar=True,
            sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", acc, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )