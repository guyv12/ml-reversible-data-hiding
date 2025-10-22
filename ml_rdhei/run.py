import torch


def init_all() -> None:
    set_device()

def set_device() -> None:
    global dev
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(torch.version.cuda) # sanity check
    torch.multiprocessing.set_start_method('spawn', force=True) # sad but necessary
    