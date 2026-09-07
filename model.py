import logging
import torch
from torch import nn


class ECGNet(nn.Module):
    def __init__(self, in_channels=12, kernel_size=7, base_width=32):
        super(ECGNet, self).__init__()
        logger = logging.getLogger(__name__)

        assert kernel_size % 2 == 1, "kernel size should be odd"
        # if kernel_size % 2 != 1:
        #     logger.warning(f"kernel size should be odd (k={kernel_size})") 
        self.padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=base_width, 
                               kernel_size=kernel_size, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(base_width)

        self.conv2 = nn.Conv1d(base_width, base_width * 2, 
                               kernel_size=kernel_size, padding=self.padding)
        self.bn2 = nn.BatchNorm1d(base_width * 2)

        self.conv3 = nn.Conv1d(base_width * 2, base_width * 4, 
                               kernel_size=kernel_size, padding=self.padding)
        self.bn3 = nn.BatchNorm1d(base_width * 4)

        self.pool = nn.MaxPool1d(2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_width * 4, 5)

    def forward(self, x):
        # 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        #2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool(x)

        #3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# for n in (12, 6, 2, 1):
#     m = ECGNet(in_channels=n)
#     out = m(torch.randn(4, n, 1000))
#     print(n, out.shape, sum(p.numel() for p in m.parameters()))