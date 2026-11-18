from collections.abc import Sequence
from dataclasses import dataclass

import pytest
import torch

from sasrec2023.base import ScoringRecommender
from sasrec2023.config import ModelConfig
from sasrec2023.data import History, Split

N_ITEMS = 6


@dataclass(frozen=True)
class FixedConfig(ModelConfig):
    pass


class Fixed(ScoringRecommender):
    """Item j scores j for every user, so the ranking is known in advance."""

    name = "base_fixed"
    config_cls = FixedConfig

    def __init__(self, config: FixedConfig | None = None) -> None:
        self.config = config or FixedConfig()

    def fit(self, split: Split) -> None:
        pass

    def score_all(
        self, users: Sequence[int], histories: Sequence[History]
    ) -> torch.Tensor:
        return torch.arange(N_ITEMS, dtype=torch.float32).expand(len(users), -1)


class WrongShape(Fixed):
    name = "base_wrong_shape"

    def score_all(
        self, users: Sequence[int], histories: Sequence[History]
    ) -> torch.Tensor:
        return torch.zeros(N_ITEMS)


def test_method_name_adds_the_variant():
    assert Fixed().method == "base_fixed"
    assert Fixed(FixedConfig(variant="nope")).method == "base_fixed_nope"


def test_recommend_ranks_by_score_and_skips_the_history():
    assert Fixed().recommend([0, 1], [(5, 3), ()], 3) == [[4, 2, 1], [5, 4, 3]]


def test_recommend_does_not_change_the_models_scores():
    model = Fixed()
    model.recommend([0], [(5,)], 2)
    assert model.score_all([0], [()])[0, 5] == 5


def test_recommend_checks_k():
    with pytest.raises(ValueError, match="between 1 and 6"):
        Fixed().recommend([0], [()], 7)
    with pytest.raises(ValueError, match="fewer than k=3"):
        Fixed().recommend([0], [(0, 1, 2, 3)], 3)


def test_score_candidates_keeps_the_candidate_order():
    candidates = torch.tensor([[2, 0, 5], [1, 4, 3]])
    scores = Fixed().score_candidates([0, 1], [(), ()], candidates)
    assert scores.tolist() == [[2.0, 0.0, 5.0], [1.0, 4.0, 3.0]]


def test_a_wrong_shape_is_reported_with_the_model_name():
    with pytest.raises(ValueError, match=r"WrongShape.score_all returned shape \(6,\)"):
        WrongShape().recommend([0], [()], 1)


def test_users_and_histories_must_match():
    with pytest.raises(ValueError, match="2 users for 1 histories"):
        Fixed().recommend([0, 1], [()], 1)


def test_a_model_without_score_all_cannot_be_built():
    class Unfinished(ScoringRecommender):
        def fit(self, split: Split) -> None:
            pass

    with pytest.raises(TypeError, match="abstract"):
        Unfinished()  # type: ignore[abstract]
