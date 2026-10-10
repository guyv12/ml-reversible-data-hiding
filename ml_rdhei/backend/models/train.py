import torch
import torch.nn as nn

from backend.models.models import save_model, load_model
from backend.data.loader import get_loader, get_dicom_loader


def data_split():
    train_loader, train_len = get_loader("../datasets/pgm/train", regex=None)
    val_loader, val_len = get_loader("../datasets/pgm/val", regex=None)
    test_loader, test_len = get_loader("../datasets/pgm/test", regex=None)

    return train_loader, val_loader, test_loader


def dicom_data_split():
    train_loader, train_len = get_dicom_loader("../datasets/dcm/train", regex=None)
    val_loader, val_len = get_dicom_loader("../datasets/dcm/val", regex=None)
    test_loader, test_len = get_dicom_loader("../datasets/dcm/test", regex=None)

    return train_loader, val_loader, test_loader


def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, num_epochs: int = 10) -> None:
    torch.manual_seed(0)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        val_epoch_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {val_epoch_loss:.4f}")

    save_model(model, "mdl.pth")
