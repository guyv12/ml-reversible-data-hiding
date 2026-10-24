import torch
import torch.nn as nn
import torch.nn.functional as fn
import argparse

from pathlib import Path


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


class ParDilatedCCNP(nn.Module):
    """
    3 parallel convs with different dilations
    concat
    res blocks
    output conv
    """

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


class SeqDilatedCNNP(nn.Module):
    """
    input conv
    concat
    res blocks
    output conv
    """
    def __init__(self, channels: int, kernel_size: int, in_channels: int = 1, out_channels: int = 1, 
                 block_num: int = 3, use_dilation: bool = True):
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


def main():
    torch.manual_seed(0) # Repro
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser(description="Create base models")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "par_dilated_3b_cnn": ParDilatedCCNP(
            channels=args.channels,
            kernel_size=args.kernel_size,
            block_num=3,
            use_dilation=True,
        ),
        "par_non_dilated_3b_cnn": ParDilatedCCNP(
            channels=args.channels,
            kernel_size=args.kernel_size,
            block_num=3,
            use_dilation=False,
        ),
        "seq_non_dilated_3b_cnn": SeqDilatedCNNP(
            channels=args.channels,
            kernel_size=args.kernel_size,
            block_num=3,
            use_dilation=False,
        ),
        "seq_non_dilated_1b_cnn": SeqDilatedCNNP(
            channels=args.channels,
            kernel_size=args.kernel_size,
            block_num=1,
            use_dilation=False,
        ),
    }

    for name, model in models.items():
        path = args.save_dir / f"{name}.pth"
        torch.save(model, path)
        print(f"Created {name}: {path}")

if __name__ == '__main__':
    main()
