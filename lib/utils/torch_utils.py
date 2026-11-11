from __future__ import annotations

import torch


def configure_torch(num_threads: int, memory_fraction: float, device: torch.device) -> None:
    torch.set_num_threads(num_threads)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
