import torch
from data.test import test_dataloader
from ml_rdhei.predictor.train_sklearn import train_kernel



def main() -> None:
    train_kernel(9)


if __name__ == "__main__":
    main()
