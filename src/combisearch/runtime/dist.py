"""Helpers for distributed (DDP) launches.

Every helper here is safe to call in single-process mode: if
``torch.distributed`` is not initialized, ``get_rank``/``get_local_rank``
return 0 and ``is_main_process`` returns ``True``.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Global rank, or 0 in single-process mode."""
    if _dist_ready():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """GPU index for this process within the node."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    """True on rank 0 (or in single-process mode)."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all ranks. No-op in single-process mode."""
    if _dist_ready():
        dist.barrier()


def device_for_local_rank() -> str:
    """Pick the CUDA device this process should own.

    Returns ``cuda:LOCAL_RANK`` when CUDA is available, otherwise ``cpu``.
    """
    if torch.cuda.is_available():
        return f"cuda:{get_local_rank()}"
    return "cpu"
