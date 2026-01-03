import torch


def main() -> None:

    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # elevate it sometime soon

    

if __name__ == "__main__":
    main()
