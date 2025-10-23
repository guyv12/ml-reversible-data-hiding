import torch

def set_device() -> None:
    global dev
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_mask(new_mask: torch.Tensor | None = None) -> None:
    """
    Sets a global mask to be used to access reference pixels.
    If none is provided it will default to ::2::2 512x512 checkerboard.
    """

    global mask

    if new_mask is None:
        mask = torch.zeros((512, 512), dtype=torch.bool).to(dev) # !GS: assumes grayscale .pgm
        mask[::2, ::2] = True

    else:
        mask = new_mask.to(dev)


set_device()
set_mask()

print(f"data package initialized: Device set to {dev}")
