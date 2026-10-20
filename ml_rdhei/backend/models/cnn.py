import torch
import torch.nn as nn
import torch.nn.functional as fn


class DoubleConv(nn.Module):
    
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
        out = fn.relu(self.conv1(X))
        return self.conv2(out)


class ResidualBlock(DoubleConv):
    """Double Convolution with skip connection"""

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return fn.relu(super().forward(X) + X)


def make_residual_blocks(channels: int, kernel_size: int, block_num: int,
                         dilations: list[int] | None = None) -> list[nn.Module]:
    if dilations is None:
        dilations = [1] * block_num
    if len(dilations) != block_num:
        raise ValueError("Provide exactly one dilation per residual block")

    return [
        ResidualBlock(channels, dilation, kernel_size)
        for dilation in dilations
    ]


class ParallelFusionBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, dilations: list[int], kernel_size: int):
        super().__init__()

        assert kernel_size % 2 == 1
        self.branches = nn.ModuleList([
            nn.Conv2d(
                in_channels, channels,
                kernel_size=kernel_size,
                padding=dilation * (kernel_size - 1) // 2,
                dilation=dilation,
            )
            for dilation in dilations
        ])

        self.fusion = nn.Conv2d(
            channels * len(dilations), channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [fn.relu(branch(X)) for branch in self.branches],
            dim=1,
        )
        return fn.relu(self.fusion(features))


class SeqDilatedCNNP(nn.Module):
    def __init__(self, channels: int, kernel_size: int, in_channels: int = 1, out_channels: int = 1, 
                 block_num: int = 3, use_dilation: bool = False):
        super().__init__()

        self.input_conv = nn.Conv2d(
            in_channels, channels, kernel_size, padding=(kernel_size - 1) // 2
        )

        self.blocks = nn.Sequential(
            *make_residual_blocks(
                channels,
                kernel_size,
                block_num,
                [i + 1 for i in range(block_num)] if use_dilation else None,
            )
        )

        self.output_conv = nn.Conv2d(
            channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = fn.relu(self.input_conv(X))
        X = self.blocks(X)
        return self.output_conv(X)


class ParDilatedCCNP(nn.Module):
    def __init__(self, channels: int, kernel_size: int, in_channels: int = 1, out_channels: int = 1, 
                 block_num: int = 3, use_dilation: bool = True):
        super().__init__()

        dilations = [1, 2, 3] if use_dilation else [1, 1, 1]
        self.input_op = ParallelFusionBlock(in_channels, channels, dilations, kernel_size)

        self.blocks = nn.Sequential(
            *make_residual_blocks(channels, kernel_size, block_num)
        )

        self.output_conv = nn.Conv2d(
            channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = fn.relu(self.input_op(X))
        X = self.blocks(X)
        return self.output_conv(X)
