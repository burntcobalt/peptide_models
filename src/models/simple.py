import torch
from torch import nn


class DenseNet(nn.Module):
    """
    simple 3 layer network
    """
    def __init__(self, config_in):
        super(DenseNet, self).__init__()
        self.config = config_in
        bins = int(self.config.max_mz / self.config.bin_size)
        self.layer1 = nn.Sequential(
            nn.Linear(420, bins),
            nn.BatchNorm1d(bins),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(bins, bins),
            nn.BatchNorm1d(bins),
            nn.ReLU())
        self.fc = nn.Linear(bins, bins)

    def forward(self, x):
        # x = torch.reshape(x, (-1, 420))
        x = x.float()
        out = torch.flatten(x, start_dim=1)  # 20x21
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.fc(out)
        return out
