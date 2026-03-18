from pathlib import Path

import click
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from nn.capsnet import CapsNet
from nn.dataloader import get_dataloaders
from nn.training import train


@click.command()
@click.option("--data", default="galaxy_images", type=click.Path(exists=True, path_type=Path))
@click.option("--outdir", default="nn_runs", type=click.Path(path_type=Path))
@click.option("--epochs", default=50, type=int)
@click.option("--batch-size", default=16, type=int)
@click.option("--lr", default=5e-4, type=float)
@click.option("--weight-decay", default=0.01, type=float)
@click.option("--train-ratio", default=0.8, type=float)
@click.option("--seed", default=42, type=int)
def main(
    data: Path,
    outdir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    train_ratio: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(
        data, batch_size=batch_size, train_ratio=train_ratio, seed=seed
    )

    model = CapsNet(
        in_ch=5,
        n_classes=2,
        primary_spatial=24 * 24,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    outdir.mkdir(parents=True, exist_ok=True)

    train(
        model, train_loader, val_loader, criterion, optimizer, device, epochs, scheduler
    )

    torch.save(model.state_dict(), outdir / "model.pt")


if __name__ == "__main__":
    main()
