import torch
import torch.nn as nn
import time
import logging
import argparse
from pathlib import Path

from backend.models.models import save_model, load_model
from backend.predictor.features import mask_batch
from backend.predictor.pipelines import reference_mask
from backend.data.loader import get_loader, get_dicom_loader
import backend.models.cnn as cnn


def data_split():
    train_loader, train_len = get_loader("../datasets/pgm/train", batch_size=8, num_workers=0)
    val_loader, val_len = get_loader("../datasets/pgm/val", batch_size=8, shuffle=False, num_workers=0)
    test_loader, test_len = get_loader("../datasets/pgm/test", batch_size=8, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def dicom_data_split():
    train_loader, train_len = get_dicom_loader("../datasets/dcm/train")
    val_loader, val_len = get_dicom_loader("../datasets/dcm/val", shuffle=False, num_workers=0)
    test_loader, test_len = get_dicom_loader("../datasets/dcm/test", shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def setup_logger(log_path: str | Path) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def train_model(model: nn.Module, path: Path | str,
                mask: torch.Tensor, num_epochs: int,
                criterion: nn.Module, optimizer: torch.optim.Optimizer,
                train_loader: torch.utils.data.DataLoader,
                val_loader: torch.utils.data.DataLoader):
    """_summary_

    Args:
        model (nn.Module): model to train
        num_epochs (int): training epochs
        criterion (nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        train_loader (torch.utils.data.DataLoader): _description_
        val_loader (torch.utils.data.DataLoader): _description_
    """
    logger = setup_logger("../logs/train.log")
    logger.info("Training Starts")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    model = model.to(device)
    mask = mask.to(device)

    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        start = time.perf_counter()

        # 1. Train loop
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            # 1. Prepare
            batch = batch.to(device=device, dtype=torch.float32) / 255.0 # Normalize (batch dtype will be uint8)

            masked_batch, non_ref_pixels, _ = mask_batch(batch, mask)
            X = masked_batch.unsqueeze(1) # transform to (B, C, H, W)
            y = non_ref_pixels

            # 2. Fw pass
            y_pred = model(X)
            y_pred = y_pred.squeeze(1) # transform back to (B, H, W)
            loss = criterion(y_pred[:, ~mask], y)

            # 3. Bw pass
            optimizer.zero_grad()
            loss.backward()

            # 4. Update weights
            optimizer.step()

            # 5. Info
            running_loss += loss.item()
            print(f"train batch {i + 1}")
        
        # 2. Validation loop
        model.eval()
        val_running_loss = 0.0

        with torch.inference_mode():
            for i, batch in enumerate(val_loader):
                # 1. Prepare
                batch = batch.to(device=device, dtype=torch.float32) / 255.0 # Normalize (batch dtype will be uint8)

                masked_batch, non_ref_pixels, _ = mask_batch(batch, mask)
                X = masked_batch.unsqueeze(1)
                y = non_ref_pixels

                # 2. Fw pass
                y_pred = model(X)
                y_pred = y_pred.squeeze(1)
                loss = criterion(y_pred[:, ~mask], y)

                # 3. Info
                val_running_loss += loss.item()
                print(f"validation batch {i + 1}")

        # 3. Info
        train_loss = running_loss / len(train_loader)
        val_loss = val_running_loss / len(val_loader)
        if device.type == 'cuda':
            torch.cuda.synchronize() # to sync time for perf_counter
        duration = time.perf_counter() - start

        logger.info(
            "Epoch [%d/%d] | Train Loss: %.6f | Validation Loss: %.6f | Duration: %.1f s (%.2f min)",
            epoch + 1, num_epochs, train_loss, val_loss, duration, duration / 60,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, path)

    #save_model(model, path)


def main():
    torch.manual_seed(0) # Repro
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser(description="Train the pixel predictor")

    parser.add_argument("--save-path", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)

    args = parser.parse_args()

    model = load_model(args.model)
    mask = reference_mask(512, 512)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader, val_loader, _ = data_split()

    train_model(
        model=model,
        path=args.path,
        mask=mask,
        num_epochs=args.epochs,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
    )


if __name__ == '__main__':
    main()