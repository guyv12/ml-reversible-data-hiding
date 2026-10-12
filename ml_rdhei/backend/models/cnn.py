import torch
import torch.nn as nn
import torch.nn.functional as fn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, # in == out for skip connection
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2, # dilation changes effective kernel size
            dilation=dilation,
        )
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        skip_connection = X

        out = fn.relu(self.conv1(X))
        out = self.conv2(out)

        return fn.relu(out + skip_connection)


class CNNPredictor(nn.Module):
    def __init__(self, channels: int, kernel_size: int, in_channels: int = 1, out_channels: int = 1, 
                 block_num: int = 3, use_dilation: bool = False):
        super().__init__()

        self.input_conv = nn.Conv2d(
            in_channels, channels, kernel_size, padding=(kernel_size - 1) // 2
        )

        self.blocks = nn.Sequential(
            *self._make_res_blocks(channels, kernel_size, block_num, use_dilation)
        )

        self.output_conv = nn.Conv2d(
            channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = fn.relu(self.input_conv(X))
        X = self.blocks(X)
        return self.output_conv(X)


    @staticmethod
    def _make_res_blocks(channels: int, kernel_size: int, block_num: int, use_dilation: bool = False) -> list[nn.Module]:
        blocks = []
        for i in range(block_num):
            dilation = i + 1 if use_dilation else 1
            blocks.append(ResidualBlock(channels, dilation, kernel_size))
        
        return blocks
