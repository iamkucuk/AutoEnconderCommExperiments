import pytorch_lightning as pl
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from DataGen import DataGen


class AEN(pl.LightningModule):

    def __init__(self, M=16, SNR=7, n_channels=7):
        super(AEN, self).__init__()
        # self.curve = []
        self.SNR_vec = np.arange(-5, 8, .5)

        self.M = M
        in_channels = int(np.log2(M))
        self.ebno = 10 ** (SNR / 10)

        self.in_channels = in_channels

        self.rate = in_channels / n_channels

        self.encoder = nn.Sequential(
            nn.Linear(M, M),
            nn.ReLU(inplace=True),
            nn.Linear(M, n_channels)
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_channels, M),
            nn.ReLU(inplace=True),
            nn.Linear(M, M)
        )

        self.normalization = nn.BatchNorm1d(n_channels)

    def AWGN(self, x, SNR):
        if SNR is None:
            ebno = self.ebno
        else:
            ebno = 10 ** (SNR / 10)
        noise = torch.randn(*x.size()) / ((2 * self.rate * ebno) ** 0.5)

        return x + noise

    def forward(self, inputs, SNR=None):
        x = self.encoder(inputs)
        x = self.normalization(x)
        x = self.AWGN(x, SNR)
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        # snr_dict = {}
        curve = []
        for idx, SNR in enumerate(self.SNR_vec):
            y_hat = self.forward(x, SNR)
            pred = y_hat.data.numpy()
            label = y.data.numpy()
            pred_output = np.argmax(pred, 1)
            error_count = (pred_output != label).astype(int).sum()
            curve.append(error_count / self.test_size)
        return curve

    def test_end(self, outputs):
        outputs_np = np.array([np.array(xi) for xi in outputs])
        curve = np.mean(outputs_np, 0)
        tensorboard_logs = {'test_ber_curve': curve}
        plt.plot(self.SNR_vec, curve)
        plt.yscale("log")
        plt.grid()
        plt.show()
        return {'avg_test_loss': curve, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.001)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        dataset = DataGen(self.M, 8000)
        return DataLoader(dataset, batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        self.test_size = 10000
        dataset = DataGen(self.M, self.test_size)
        return DataLoader(dataset, batch_size=32)
