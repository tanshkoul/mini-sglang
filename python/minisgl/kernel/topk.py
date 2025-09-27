from __future__ import annotations

from typing import Any

import torch

from .utils import load_kernel_module


def _load_topk_module() -> Any:
    """
    Load the index manipulation module.
    """
    return load_kernel_module("topk.cu", "topk_kernel")


def fast_topk(
    score: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    return _load_topk_module().fast_topk(score, indices, lengths)
