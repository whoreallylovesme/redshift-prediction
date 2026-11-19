from __future__ import annotations

import lightning as L
import torch

from lib.models.capsnet import CapsNet
from photoz.targets import logistic_to_z


class CapsNetPredictor(L.LightningModule):
    def __init__(
        self,
        weights: str,
        in_channels: int,
        input_size: int,
        z_min: float,
        z_max: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.model = CapsNet(
            in_channels=in_channels,
            input_size=input_size,
            num_class=2,
            dim_capsule=16,
            routings=3,
        ).to(device)
        self.model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        self.model.eval()
        self.z_min = z_min
        self.z_max = z_max
        self.to(device)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = images.to(self.device, non_blocking=True)
        z_logistic, probs, _ = self.model(images, target_class=None)
        return {
            "z_pred": logistic_to_z(z_logistic, self.z_min, self.z_max),
            "p_spiral": probs[:, 1],
        }

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        return self.predict(batch[0])
