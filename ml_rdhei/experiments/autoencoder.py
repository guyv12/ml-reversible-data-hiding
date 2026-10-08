import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import ml_rdhei.backend.data.loader as dloader
import ml_rdhei.experiments.dataset as ddataset
import ml_rdhei.backend.compressor.compress as compressor

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(32*5*5, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 32*5*5),
            nn.ReLU(),

            nn.Unflatten(1, (32, 5, 5)),
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)

        return reconstructed

def test_shapes():
    model = Autoencoder()
    x = torch.randn(8, 2, 5, 5)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    latent = model.encoder(x)
    print(f"Latent shape: {latent.shape}")

def test_training():
    model = Autoencoder()

    x = torch.rand(1, 2, 5, 5)
    target = x[:, 0:1, :, :]

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range (1000):
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item():.6f}")

    with torch.no_grad():
        output = model(x)

    print(f"Target:\n{target[0, 0]}")
    print(f"\nReconstructed:\n{output[0, 0]}")


set, _ = dloader.get_loader("datasets\BOSSbase_512", batch_size=1, num_workers=0)
image = next(iter(set)).reshape(512, 512)
image = image / 255.0

dataset = ddataset.PixelMaskDataset(image)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

def training():
    model = Autoencoder()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 10

    for epoch in range(epochs):
        total_loss = 0.0

        for input_batch, target_batch in loader:
            optimizer.zero_grad()

            output = model(input_batch)

            prediction = output[:, 0, 2, 2]

            loss = criterion(prediction, target_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch: {epoch + 1}/{epochs}, "
            f"Loss: {average_loss:.6f}"
        )

    torch.save(model, "autoencoder_model.pth")

def eval(model):
    model.eval()

    test_image = next(iter(set)).reshape(512, 512)
    test_image = test_image / 255.0
    test_dataset = ddataset.PixelMaskDataset(test_image)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

    errors = torch.zeros(len(test_dataset), dtype=torch.int64)

    total_absolute_error = 0.0
    total_squared_error = 0.0
    total_samples = 0

    max_error = 0
    exact_predictions = 0
    sample_offset = 0

    with torch.no_grad():
        for input_batch, target_batch in test_loader:
            output = model(input_batch)

            prediction = output[:, 0, 2, 2]

            prediction_pixels = torch.round(prediction * 255)
            target_pixels = torch.round(target_batch * 255)

            error = target_pixels - prediction_pixels

            batch_end = sample_offset + error.numel()
            errors[sample_offset:batch_end] = error
            sample_offset = batch_end

            total_absolute_error += torch.abs(error).sum().item()
            total_squared_error += (error ** 2).sum().item()

            max_error = max(
                max_error,
                torch.abs(error).max().item()
            )

            exact_predictions += (error == 0).sum().item()
            total_samples += error.numel()

    mae = total_absolute_error / total_samples
    rmse = (total_squared_error / total_samples) ** 0.5
    exact_rate = exact_predictions / total_samples * 100

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Maximum absolute error: {max_error:.0f}")
    print(f"Exact prediction rate: {exact_rate:.2f}%")

    error_map = torch.zeros((test_dataset.height, test_dataset.width), dtype=torch.int64)
    for idx, (row, col) in enumerate(test_dataset.coords):
        error_map[row, col] = errors[idx]

    #print(f"Error map: {error_map[:16, :16]}")
    #compressor.__compress_error_map(error_map)
    
training()
eval(torch.load("autoencoder_model.pth", weights_only=False))
#test_shapes()
#test_training()
