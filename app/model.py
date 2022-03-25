import numpy as np
import torch
from torch import nn
from torch.nn import L1Loss, BCEWithLogitsLoss
import pytorch_lightning as pl
from tcn import TemporalConvNet


class TCN(pl.LightningModule):
    def __init__(
        self,
        vocab,
        num_size,
        output_size,
        num_channels,
        kernel_size,
        dropout,
        learning_rate,
        model_type,
    ):
        super(TCN, self).__init__()
        input_size = int(
            np.sum([int(np.floor(len(vocab[c]) / 4) + 1) for c in vocab]) + num_size
        )
        if model_type == "regressor":
            self.criterion = L1Loss()
            self.output_last = False
        else:
            self.criterion = BCEWithLogitsLoss()
            self.output_last = True
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    len(vocab[c]), int(np.floor(len(vocab[c]) / 4) + 1), padding_idx=0
                )
                for c in vocab
            ]
        )
        self.learning_rate = learning_rate
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.softplus = nn.Softplus()

    def forward(self, x1, x2):
        # Embed each categorical variable separately.
        # Expects input (batch, sequence, categorical variable).
        xs = [
            self.embeddings[i](x2[:, :, i].long()) for i in range(len(self.embeddings))
        ]
        # concat then transpose for convolution
        x2 = torch.cat(xs, dim=2).transpose(1, 2)
        x = torch.cat([x1, x2], dim=1).float()
        y1 = self.tcn(x)  # input should have dimension (N, C, L)
        if self.output_last:
            o = self.linear(y1[:, :, -1])
        else:
            o = self.softplus(self.linear(y1.transpose(1, 2))).squeeze(-1)
        return o

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_loss(self, batch):
        x1, x2, target = batch
        x1, target = (x1.transpose(1, 2), target.squeeze(-1))
        output = self(x1, x2)
        loss = self.criterion(output, target)
        return loss

    def training_step(self, batch, batch_idx):
        train_loss = self.compute_loss(batch)
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.compute_loss(batch)
        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss = self.compute_loss(batch)
        self.log(
            "test_loss",
            test_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return test_loss
