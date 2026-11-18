import random

import torch

from sasrec2023.decorators import seeded


def draw() -> tuple[float, list[float]]:
    return random.random(), torch.rand(3).tolist()


def test_same_seed_same_numbers():
    with seeded(5):
        first = draw()
    with seeded(5):
        second = draw()
    assert first == second


def test_different_seeds_different_numbers():
    with seeded(5):
        first = draw()
    with seeded(6):
        second = draw()
    assert first != second


def test_the_outside_sequence_continues_as_if_nothing_ran():
    random.seed(0)
    torch.manual_seed(0)
    expected = [draw(), draw()]

    random.seed(0)
    torch.manual_seed(0)
    got = [draw()]
    with seeded(99):
        draw()
    got.append(draw())
    assert got == expected
