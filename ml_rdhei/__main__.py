import torch
from data.test import test_dataloader
from predictor.train import train_kernel


def main() -> None:

    train_kernel()


if __name__ == "__main__":
    main()
