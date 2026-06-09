"""Global seed fixing for reproducible CombiSearch runs.

Every CLI entrypoint should call :func:`set_global_seed` once, as early as
possible, so that all downstream randomness (Python ``random``, NumPy,
PyTorch CPU/CUDA, cuDNN, HuggingFace ``set_seed``) is pinned to the same
value.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_SEED = 0
_SEED_ENV_VAR = "COMBISEARCH_SEED"


def resolve_seed(runtime_config: dict[str, Any] | None = None,
                 default: int = DEFAULT_SEED) -> int:
    """Resolve the active seed.

    Precedence: ``runtime_config['seed']`` > ``$COMBISEARCH_SEED`` > ``default``.
    """
    if runtime_config is not None and runtime_config.get("seed") is not None:
        return int(runtime_config["seed"])
    env_value = os.environ.get(_SEED_ENV_VAR)
    if env_value is not None and env_value != "":
        return int(env_value)
    return int(default)


def set_global_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> int:
    """Pin every known source of randomness to ``seed``.

    Sets Python ``random``, NumPy, PyTorch (CPU + all CUDA devices),
    cuDNN flags, ``PYTHONHASHSEED``, and HuggingFace ``transformers.set_seed``
    when available. Returns the seed for convenience.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    try:
        from transformers import set_seed as hf_set_seed
    except ImportError:
        hf_set_seed = None
    if hf_set_seed is not None:
        hf_set_seed(seed)

    logger.info("Global seed set to %d (deterministic=%s)", seed, deterministic)
    return seed
