"""Fixtures shared by the model tests: small random data, and a guard for leaks."""

import random

import pytest

from sasrec2023.data import Split

N_USERS = 60
N_ITEMS = 30


class NoTestSplit(Split):
    """A Split whose test cases cannot be read: `fit` must never need them."""

    def test(self) -> tuple[list[tuple[int, ...]], list[int]]:
        raise AssertionError("fit() read split.test(): the test answers leak")


def make_split(cls: type[Split] = Split, seed: int = 0) -> Split:
    """60 users with 5 to 12 distinct items each, out of 30 items."""
    rng = random.Random(seed)
    sequences = tuple(
        tuple(rng.sample(range(N_ITEMS), rng.randint(5, 12))) for _ in range(N_USERS)
    )
    users = tuple(f"user{u}" for u in range(N_USERS))
    items = tuple(f"item{j}" for j in range(N_ITEMS))
    return cls(sequences, users, items)


@pytest.fixture
def fake_split() -> Split:
    return make_split()


@pytest.fixture
def no_test_split() -> Split:
    return make_split(NoTestSplit)
