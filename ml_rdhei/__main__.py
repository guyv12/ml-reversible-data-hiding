import torch
from data.test import test_dataloader


def main() -> None:

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # elevate it sometime soon

    for i in range(10):
        test_dataloader(dev)


if __name__ == "__main__":
    main()
