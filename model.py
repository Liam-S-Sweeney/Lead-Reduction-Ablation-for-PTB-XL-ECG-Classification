import torch
from torch import nn


class ECGNet(nn.Module):
    """
    1D CNN for multi-label ECG superclass classification.
    
    Stacks `num_layers` blocks of Conv1d -> BatchNorm1d -> ReLU -> MaxPool1d(2).
        - Channel width starts at `base_width` and doubles each block
        - Sequence length halves each block
        - Global average pooling collapses the time axis, and a linear layer maps to 5 logits 
            - (CD, HYP, MI, NORM, STTC).
    Outputs raw logits, not probabilities: use with BCEWithLogitsLoss and apply a sigmoid for inference.
    
    Args:
        in_channels: number of input leads (12 for a full ECG; fewer in the ablation)
        kernel_size: convolution width in samples; must be odd so padding
                    preserves length (7 samples = 70 ms at 100 Hz)
        base_width:  output channels of the first block
        num_layers:  number of conv blocks; each doubles width and halves length

    Input shape:  (batch, in_channels, 1000)
    Output shape: (batch, 5)
    """
    def __init__(self, in_channels=12, kernel_size=7, base_width=32, num_layers=3):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if 1000 // 2 ** num_layers < kernel_size:   # 1000 = input length at 100 Hz
            raise ValueError(
                f"num_layers={num_layers} reduces the signal to {1000 // 2 ** num_layers} "
                f"samples, shorter than kernel_size={kernel_size}")
        
        padding = (kernel_size - 1) // 2    # "same" padding: conv output length = input length
        
        self.layers = nn.ModuleList()   # Registered, so optimizer/.to()/checkpoints see these layers
        current_in = in_channels    # 12 (or fewer, in the ablation)
        current_out = base_width    # 32 (default)

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

        self.gap = nn.AdaptiveAvgPool1d(1)  # average each channel over time: (batch, 128, 125) -> (batch, 128, 1)
        self.fc = nn.Linear(current_in, 5)  # 5 superclasses

    def forward(self, x):
        """
        Compute class logits for a batch of ECG signals.

        Passes the input through each conv block in order, then global-average-
        pools over time, flattens, and applies the linear classifier.

        Args:
            x: tensor of shape (batch, in_channels, 1000), normalized per lead

        Returns:
            Tensor of shape (batch, 5): raw logits for CD, HYP, MI, NORM, STTC.
            Not probabilities; apply torch.sigmoid for per-class probabilities.
        """
        for layer in self.layers:
            x = layer(x)
            
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x