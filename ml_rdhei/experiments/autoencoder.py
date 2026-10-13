import argparse
import torch.nn as nn
import torch
from ml_rdhei.experiments.dataset import *
from ml_rdhei.backend.data.loader import get_loader
from ml_rdhei.backend.models.models import save_model
from ml_rdhei.backend.models.train import setup_logger

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(64*5*5, 128)
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 64*5*5),
            nn.ReLU(),

            nn.Unflatten(1, (64, 5, 5)),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        return reconstructed

def data_split():
    train_loader, train_len = get_loader("../datasets/pgm/train", batch_size=8, num_workers=0)
    val_loader, val_len = get_loader("../datasets/pgm/val", batch_size=8, shuffle=False, num_workers=0)
    test_loader, test_len = get_loader("../datasets/pgm/test", batch_size=8, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_len, val_len, test_len

def train(model, epochs, criterion, optimizer, train_loader, val_loader, train_len, val_len):

    logger = setup_logger("../logs/train.log")
    logger.info("Training Starts")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    model = model.to(device)

    best_val_loss = float("inf")

    # train loop
    for epoch in range(epochs):
        model.train()
        train_total_loss = 0.0

        for _ in range(train_len):
            image = next(iter(train_loader)).reshape(512, 512)
            train_dataset = PatchDataset(image)

            for x, target, ref_mask in train_dataset:
                optimizer.zero_grad()

                pred = model(x)[:, 0]

                loss = criterion(pred[:, ~ref_mask], target)

                loss.backward()
                optimizer.step()

                train_total_loss += loss.item()

        train_loss = train_total_loss / len(train_loader)

        print(
            f"Epoch: {epoch + 1}/{epochs}, "
            f"Loss: {train_loss:.6f}"
        )


        # validation loop
        model.eval()
        val_total_loss = 0.0


        with torch.no_grad():
            for _ in range(val_len):
                image = next(iter(val_loader)).reshape(512, 512)
                val_dataset = PatchDataset(image)

                for x, target, ref_mask in val_dataset:
                    pred = model(x)[:, 0]

                    loss = criterion(pred[:, ~ref_mask], target)
                    val_total_loss += loss.item()


        val_loss = val_total_loss / len(val_loader)
        print(
            f"Epoch: {epoch + 1}/{epochs}, "
            f"Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, "autoencoder.pth")

def main():
    torch.manual_seed(0) # Repro
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser(description="Train the pixel predictor")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)

    args = parser.parse_args()

    # Model
    model = Autoencoder(
        channels=args.channels,
        kernel_size=args.kernel_size,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader, val_loader, test_loader, train_len, val_len, test_len = data_split()

    train(
        model=model,
        epochs=args.epochs,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        train_len=train_len,
        val_len=val_len,
    )

if __name__ == "__main__":
    main()
