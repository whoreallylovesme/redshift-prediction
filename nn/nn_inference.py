from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from nn.capsnet import CapsNet


def load_npz(path: Path) -> torch.Tensor:
    data = np.load(path)
    img = data["image"].astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img


@click.command()
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--data", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", default="preds.parquet", type=click.Path(path_type=Path))
@click.option("--batch-size", default=32, type=int)
def main(model: Path, data: Path, out: Path, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CapsNet(in_ch=5, n_classes=2, primary_spatial=24 * 24)
    net.load_state_dict(torch.load(model, map_location=device, weights_only=True))
    net.to(device)
    net.eval()

    root = Path(data)
    paths = sorted(root.rglob("*.npz"))
    if not paths:
        raise ValueError(f"No .npz files found in {data}")

    preds_list: list[dict] = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch = torch.cat([load_npz(p) for p in batch_paths], dim=0).to(device)
        with torch.no_grad():
            logits = net(batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        for path, pred, prob in zip(
            batch_paths, preds.cpu().numpy(), probs.cpu().numpy(), strict=True
        ):
            objid = path.stem
            preds_list.append({
                "objid": objid,
                "pred": int(pred),
                "prob_teddy": float(prob[0]),
                "prob_happy": float(prob[1]),
            })

    df = pd.DataFrame(preds_list)
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} predictions to {out}")


if __name__ == "__main__":
    main()
