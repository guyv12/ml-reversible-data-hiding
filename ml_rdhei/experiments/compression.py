import argparse
import math
import time
import torch
import torch.nn as nn

from backend.data.loader import get_loader
from backend.models.models import save_model
from backend.models.train import setup_logger


DEFAULT_LATENT_DIM = 1024
DEFAULT_INPUT_SIZE = 65536

class ResidualBlock(nn.Module):
    """Two 3x3 convs with an identity shortcut (skip connection inside encoder/decoder)."""
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


def _down(in_ch: int, out_ch: int) -> nn.Sequential:
    # learned stride-2 downsampling (H, W) -> (H/2, W/2)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        ResidualBlock(out_ch),
    )


def _up(in_ch: int, out_ch: int) -> nn.Sequential:
    # resize + conv instead of ConvTranspose2d to reduce checkerboard artifacts
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(),
        ResidualBlock(out_ch),
    )


class CompressionAutoencoder(nn.Module):
    """Fully convolutional autoencoder for the (side x side) reference-pixel grid.

    The latent is a (latent_channels, side/32, side/32) tensor, exposed as a flat
    vector of length latent_dim, e.g. 1024 = 16 x 8 x 8 for a 256x256 grid.
    """
    N_DOWN = 5

    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM, input_size: int = DEFAULT_INPUT_SIZE):
        super().__init__()

        side = math.isqrt(input_size)
        if side * side != input_size or side % 2 ** self.N_DOWN != 0:
            raise ValueError(f"input_size must be a square side^2 with side divisible by {2 ** self.N_DOWN}")

        self.latent_hw = side // 2 ** self.N_DOWN
        if latent_dim % (self.latent_hw ** 2) != 0:
            raise ValueError(f"latent_dim must be divisible by {self.latent_hw ** 2}")

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.side = side
        self.latent_channels = latent_dim // self.latent_hw ** 2

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),         # 256
            nn.ReLU(),
            _down(32, 64),                                      # 128
            _down(64, 128),                                     # 64
            _down(128, 128),                                    # 32
            _down(128, 128),                                    # 16
            _down(128, 128),                                    # 8
            nn.Conv2d(128, self.latent_channels, kernel_size=3, padding=1),  # latent, no activation
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.latent_channels, 128, kernel_size=3, padding=1),  # 8
            nn.ReLU(),
            ResidualBlock(128),
            _up(128, 128),                                      # 16
            _up(128, 128),                                      # 32
            _up(128, 64),                                       # 64
            _up(64, 32),                                        # 128
            _up(32, 32),                                        # 256
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        z = latent.reshape(-1, self.latent_channels, self.latent_hw, self.latent_hw)
        return self.decoder(z).flatten(1)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        latent = self.encode(x)
        reconstructed = self.decode(latent)

        if return_latent:
            return reconstructed, latent

        return reconstructed

def data_split(image_batch_size: int):
    train_loader, _ = get_loader("../datasets/pgm/train", batch_size=image_batch_size, num_workers=0)
    val_loader, _ = get_loader("../datasets/pgm/val", batch_size=image_batch_size, shuffle=False, num_workers=0)
    test_loader, _ = get_loader("../datasets/pgm/test", batch_size=image_batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def run_epoch(model, image_loader, criterion, optimizer, device, train_mode):
    if train_mode:
        model.train()
        grad_context = torch.enable_grad()
    else:
        model.eval()
        grad_context = torch.no_grad()

    total_loss = 0.0
    total_batches = 0

    with grad_context:
        for image_batch in image_loader:
            # (B, H, W) uint8 -> (B, 1, H/2, W/2) float in [0, 1]
            input_batch = image_batch[:, ::2, ::2].unsqueeze(1).float().div(255.0).to(device)
            reference_pixels = input_batch.flatten(1)

            if train_mode:
                optimizer.zero_grad()
            
            reconstructed, _ = model(input_batch, return_latent=True)
            loss = criterion(reconstructed, reference_pixels)

            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    average_loss = total_loss / max(total_batches, 1)
    return average_loss


def train(model, epochs, criterion, optimizer, train_loader, val_loader, path):

    logger = setup_logger("../logs/train.log")
    logger.info("Training Starts")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    model = model.to(device)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        start = time.perf_counter()

        train_loss = run_epoch(
            model=model,
            image_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_mode=True,
        )
        val_loss = run_epoch(
            model=model,
            image_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_mode=False,
        )

        if device.type == 'cuda':
            torch.cuda.synchronize()
        duration = time.perf_counter() - start

        logger.info(
            "Epoch [%d/%d] | Train Loss: %.6f | Validation Loss: %.6f | Duration: %.1f s (%.2f min)",
            epoch + 1, epochs, train_loss, val_loss, duration, duration / 60,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, path)

def main():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser(description="Train the patch autoencoder")

    parser.add_argument("--path", type=str, default="compression_autoencoder.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    args = parser.parse_args()

    model = CompressionAutoencoder(
        latent_dim=args.latent_dim,
        input_size=args.input_size,
    )

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader, val_loader, _ = data_split(args.image_batch_size)

    train(
        model=model,
        epochs=args.epochs,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        path=args.path,
    )

if __name__ == "__main__":
    main()
