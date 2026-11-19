from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from photoz.data.euclid_packet import IT_ARGS, Rows, load_it_from_clip, masked
from photoz.models.euclid_clip import BinNet, ITNetwork, bin_summary
from photoz.training import euclid_clip_finetune as fine
from photoz.training import euclid_clip_pretrain as ssl


class TinySource:
    def __init__(self) -> None:
        g = torch.Generator().manual_seed(194)
        self.descriptor = torch.randn(14, 18, generator=g)
        image = torch.randn(5, 12, 64, 64, generator=g)
        image[:, 8:] = 1
        self.arrays = {
            "image": image,
            "image_available": torch.ones(5, dtype=torch.bool),
            "table_values": torch.randn(5, 14, 2, generator=g),
            "table_mask": torch.ones(5, 14, dtype=torch.bool),
            "table_available": torch.ones(5, dtype=torch.bool),
            "spectrum": torch.randn(5, 8, 4, generator=g),
            "spectrum_mask": torch.ones(5, 8, dtype=torch.bool),
            "spectrum_metadata": torch.randn(5, 3, generator=g),
            "spectrum_available": torch.ones(5, dtype=torch.bool),
            "field_code": torch.ones(5, dtype=torch.int64),
            "object_id": torch.arange(5),
        }

    def batch(
        self, rows: Rows, device: torch.device, spectra: bool = True
    ) -> dict[str, torch.Tensor]:
        data = {k: a[torch.as_tensor(rows)].to(device) for k, a in self.arrays.items()}
        data["table_descriptor"] = self.descriptor.to(device)
        return data


class ClipTrainingTests(TestCase):
    def test_real_encoders_pretrain_and_finetune_from_exported_weights(self) -> None:
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)
        data = TinySource()
        device = torch.device("cpu")
        with patch.object(ssl, "SSL_BATCH", 2):
            model, history = ssl.pretrain(ITNetwork(), data, np.arange(3), 77, 6, device)
        for record in history:
            self.assertEqual(record["examples"], 2)
            self.assertEqual(record["pair_examples"], {"I__S": 2, "I__T": 2, "S__T": 2})
            self.assertTrue(np.isfinite(record["loss"]))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoder.pt"
            torch.save(model.state_dict(), path)
            head = BinNet(load_it_from_clip(path))
            train = {k: data.arrays[k][:3].numpy() for k in fine.TRAIN_KEYS}
            valid = {k: data.arrays[k][3:].numpy() for k in fine.TRAIN_KEYS}
            cfg = fine.EuclidClipConfig.model_construct(seed=23, epochs=6)
            with patch.object(fine, "FINE_BATCH", 2):
                metrics = fine.fine_tune(
                    head,
                    train,
                    np.array([0.1, 0.2, 0.3]),
                    valid,
                    np.array([0.15, 0.42]),
                    data.descriptor,
                    cfg,
                    device,
                )
            self.assertEqual([v["epoch"] for v in metrics], [5, 6])
            self.assertTrue(all(np.isfinite(v["train_nll"]) for v in metrics))
            torch.save(head.state_dict(), path)
            restored = fine.load_finetuned(path, device)
            get = fine.array_batches(valid, data.descriptor, device)
            keep = fine.band_keep(device)
            with patch.object(fine, "INFER_BATCH", 1):
                median, width = fine.predict(restored, get, 2, keep)
            with torch.inference_mode():
                inputs = masked(get(0, 2), keep)
                logits = restored(**{name: inputs[name] for name in IT_ARGS})
                expected = bin_summary(torch.softmax(logits.float(), 1).numpy())
            np.testing.assert_allclose(median, expected[0], rtol=1e-6)
            np.testing.assert_allclose(width, expected[1], rtol=1e-6)
            self.assertEqual(median.shape, (2,))
            self.assertTrue(np.isfinite(width).all())
