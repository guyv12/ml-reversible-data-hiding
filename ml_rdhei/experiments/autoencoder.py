import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.dataset import PatchDataset
from backend.data.loader import get_loader
from backend.models.models import save_model
from backend.models.train import setup_logger


DEFAULT_PATCH_SIZE = 64
DEFAULT_LATENT_DIM = 256

class Autoencoder(nn.Module):
    def __init__(self, patch_size: int = DEFAULT_PATCH_SIZE, latent_dim: int = DEFAULT_LATENT_DIM):
        super().__init__()
        if patch_size % 8 != 0:
            raise ValueError("patch_size must be divisible by 8 to match pooling depth")

        reduced_size = patch_size // 8
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.latent_channels = 128

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, self.latent_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.to_latent = nn.Linear(self.latent_channels * reduced_size * reduced_size, latent_dim)

        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, self.latent_channels * reduced_size * reduced_size),
            nn.ReLU(),
            nn.Unflatten(1, (self.latent_channels, reduced_size, reduced_size)),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        flattened = self.flatten(features)
        return self.to_latent(flattened)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        restored = self.from_latent(latent)
        return self.decoder(restored)

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


def iter_patch_loaders(image_batch: torch.Tensor, patch_size: int, patch_batch_size: int, shuffle: bool):
    for image in image_batch:
        yield DataLoader(
            PatchDataset(image, patch_size=patch_size),
            batch_size=patch_batch_size,
            shuffle=shuffle,
        )


def run_epoch(model, image_loader, criterion, optimizer, device, patch_size, patch_batch_size, train_mode):
    if train_mode:
        model.train()
        grad_context = torch.enable_grad()
    else:
        model.eval()
        grad_context = torch.no_grad()

    total_loss = 0.0
    total_patch_batches = 0

    with grad_context:
        for image_batch in image_loader:
            for patch_loader in iter_patch_loaders(image_batch, patch_size, patch_batch_size, shuffle=train_mode):
                for input_batch, target_batch, ref_mask_batch in patch_loader:
                    input_batch = input_batch.to(device)
                    target_batch = target_batch.to(device)
                    ref_mask_batch = ref_mask_batch.to(device)

                    if train_mode:
                        optimizer.zero_grad()

                    reconstructed, _ = model(input_batch, return_latent=True)
                    prediction = reconstructed[:, 0]
                    loss = criterion(prediction[~ref_mask_batch], target_batch[~ref_mask_batch])

                    if train_mode:
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item()
                    total_patch_batches += 1

    average_loss = total_loss / max(total_patch_batches, 1)
    return average_loss


def train(model, epochs, criterion, optimizer, train_loader, val_loader, path, patch_size, patch_batch_size):

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
            patch_size=patch_size,
            patch_batch_size=patch_batch_size,
            train_mode=True,
        )
        val_loss = run_epoch(
            model=model,
            image_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            patch_size=patch_size,
            patch_batch_size=patch_batch_size,
            train_mode=False,
        )

        if device.type == 'cuda':
            torch.cuda.synchronize() # to sync time for perf_counter
        duration = time.perf_counter() - start

        logger.info(
            "Epoch [%d/%d] | Train Loss: %.6f | Validation Loss: %.6f | Duration: %.1f s (%.2f min)",
            epoch + 1, epochs, train_loss, val_loss, duration, duration / 60,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, path)


def test(model, test_loader, criterion, patch_size, patch_batch_size):
    logger = setup_logger("../logs/test.log")
    logger.info("Testing Starts")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    model = model.to(device)
    test_loss = run_epoch(
        model=model,
        image_loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        patch_size=patch_size,
        patch_batch_size=patch_batch_size,
        train_mode=False,
    )
    logger.info("Test Loss: %.6f", test_loss)

def main():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser(description="Train the patch autoencoder")

    parser.add_argument("--path", type=str, default="autoencoder.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--patch-batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--learning-rate", type=float, default=1e-3)

    args = parser.parse_args()

    model = Autoencoder(
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader, val_loader, test_loader = data_split(args.image_batch_size)

    train(
        model=model,
        epochs=args.epochs,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        path=args.path,
        patch_size=args.patch_size,
        patch_batch_size=args.patch_batch_size,
    )

    #test(
    #    model=model,
    #    test_loader=test_loader,
    #    criterion=criterion,
    #    patch_size=args.patch_size,
    #    patch_batch_size=args.patch_batch_size,
    #)

if __name__ == "__main__":
    main()
