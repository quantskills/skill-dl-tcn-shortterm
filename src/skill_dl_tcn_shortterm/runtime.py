"""Scoped runtime controls used by reproducible CPU experiments."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from .experiment import ContractError


@contextmanager
def torch_thread_scope(torch_threads: int | None) -> Iterator[int]:
    """Temporarily set PyTorch intra-op threads and restore the caller's value."""

    original = torch.get_num_threads()
    if torch_threads is None:
        yield original
        return
    if torch_threads <= 0:
        raise ContractError("torch threads must be positive")
    torch.set_num_threads(torch_threads)
    try:
        yield torch.get_num_threads()
    finally:
        torch.set_num_threads(original)
