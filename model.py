import torch
from torch import nn


class ECGNet(nn.Module):
    def __init__(self, in_channels=12, kernel_size=7, base_width=32, num_layers=3):
        super(ECGNet, self).__init__()

        assert kernel_size % 2 == 1, "kernel size should be odd"
        padding = (kernel_size - 1) // 2

        self.layers = nn.ModuleList()
        current_in = in_channels
        current_out = base_width

        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(current_in, current_out, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(current_out),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )
            )
            current_in = current_out
            current_out *= 2

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(current_in, 5)      # 5 superclasses

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x