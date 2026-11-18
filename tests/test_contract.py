"""Checks every model must pass (docs/team_spec.md, Section 13).

Add each new model to MODELS. `Reference` is a test-only model that shows the
shape of a correct model and keeps these checks running before the real
models exist.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest
import torch

from sasrec2023.base import ScoringRecommender
from sasrec2023.config import ModelConfig
from sasrec2023.data import History, Split
from sasrec2023.decorators import seeded


@dataclass(frozen=True)
class ReferenceConfig(ModelConfig):
    pass


class Reference(ScoringRecommender):
    """Random item scores drawn in `fit` from the configured seed."""

    name = "contract_reference"
    config_cls = ReferenceConfig

    def __init__(self, config: ReferenceConfig | None = None) -> None:
        self.config = config or ReferenceConfig()
        self._scores: torch.Tensor | None = None

    def fit(self, split: Split) -> None:
        split.train()
        split.valid()
        with seeded(self.config.seed):
            self._scores = torch.rand(split.n_items)

    def score_all(
        self, users: Sequence[int], histories: Sequence[History]
    ) -> torch.Tensor:
        if self._scores is None:
            raise RuntimeError("call fit() before scoring")
        return self._scores.expand(len(users), -1)


# Write new models into this list itself: pytest reads it once, when the
# fixture below is defined, so MODELS.append() further down has no effect.
MODELS: list[Callable[[], ScoringRecommender]] = [Reference]


@pytest.fixture(params=MODELS, ids=lambda make: make.__name__)
def make_model(request: pytest.FixtureRequest) -> Callable[[], ScoringRecommender]:
    make: Callable[[], ScoringRecommender] = request.param
    return make


def cases(split: Split) -> tuple[list[int], list[History]]:
    histories, _ = split.valid()
    return list(range(split.n_users)), histories


def test_fit_never_reads_the_test_cases(make_model, no_test_split):
    make_model().fit(no_test_split)


def test_one_score_per_user_and_item(make_model, fake_split):
    model = make_model()
    model.fit(fake_split)
    users, histories = cases(fake_split)
    scores = model.score_all(users, histories)
    assert scores.shape == (fake_split.n_users, fake_split.n_items)
    assert torch.isfinite(scores).all()


def test_recommendations_leave_out_the_history(make_model, fake_split):
    model = make_model()
    model.fit(fake_split)
    users, histories = cases(fake_split)
    top = model.recommend(users, histories, 10)
    assert all(
        len(items) == len(set(items)) == 10 and not set(items) & set(history)
        for items, history in zip(top, histories, strict=True)
    )


def test_same_seed_same_scores(make_model, fake_split):
    first, second = make_model(), make_model()
    first.fit(fake_split)
    second.fit(fake_split)
    users, histories = cases(fake_split)
    assert torch.equal(
        first.score_all(users, histories), second.score_all(users, histories)
    )
