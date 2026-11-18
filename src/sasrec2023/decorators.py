"""Helpers that wrap a block of code."""

import random
from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def seeded(seed: int) -> Iterator[None]:
    """Seed Python's `random` and torch for a block, then restore their states.

    Use it around a model's training, `with seeded(self.config.seed): ...`, so
    two fits with the same configuration give the same model. Restoring the
    states afterwards means one model's training does not change the random
    numbers of whatever runs next.

    Args:
        seed: Seed of both generators.
    """
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    mps = torch.backends.mps.is_available()
    mps_state = torch.mps.get_rng_state() if mps else None
    cuda = torch.cuda.is_available()
    cuda_states = torch.cuda.get_rng_state_all() if cuda else None
    random.seed(seed)
    torch.manual_seed(seed)  # also seeds MPS and CUDA
    try:
        yield
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
