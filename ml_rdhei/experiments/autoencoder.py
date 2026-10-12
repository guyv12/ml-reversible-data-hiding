import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import ml_rdhei.backend.data.loader as dloader
import ml_rdhei.experiments.dataset as ddataset
import ml_rdhei.backend.compressor.compress as compressor
from ml_rdhei.backend.models.models import save_model

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


def train(epochs, train_loader, val_loader):
    model = Autoencoder()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )
    best_val_loss = float("inf")

    # train loop
    for epoch in range(epochs):
        model.train()
        train_total_loss = 0.0

        for x, target, ref_mask in train_loader:
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
            for x, target, ref_mask in val_loader:
                pred = model(x)[:, 0]

                loss = criterion(pred[:, ~ref_mask], target)
                val_total_loss += loss.item()

                #prediction_pixels = torch.round(pred * 255)
                #target_pixels = torch.round(target * 255)


        val_loss = val_total_loss / len(val_loader)
        print(
            f"Epoch: {epoch + 1}/{epochs}, "
            f"Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, "autoencoder.pth")

def main():
    train_loader = None
    val_loader = None
    epochs = 10

    #set, _ = dloader.get_loader("datasets\BOSSbase_512", batch_size=1, num_workers=0)
    #image = next(iter(set)).reshape(512, 512)
    #dataset = ddataset.PixelMaskDataset(image)
    #loader = DataLoader(dataset, batch_size=256, shuffle=True)

    train(epochs, train_loader, val_loader)

if __name__ == "__main__":
    main()
