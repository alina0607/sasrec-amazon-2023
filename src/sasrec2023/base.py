"""The base class of every method in the study.

Every method (PopRec, Random, BPR, FMC, FPMC, SASRec, GRU4Rec) gives each item
a score and recommends the highest. A model writes `fit` and `score_all`; the
two ways of reading the scores that the evaluation needs, the 101 candidates
and the top items, are written once here, so every model is read the same way.
"""

from abc import abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import torch

from sasrec2023.config import ModelConfig
from sasrec2023.data import History, Split
from sasrec2023.registry import RecommenderMeta


class ScoringRecommender(metaclass=RecommenderMeta):
    """A model that scores every item; a higher score is recommended first.

    A subclass sets the class attributes `name` and `config_cls`, sets
    `self.config` in `__init__`, and writes `fit` and `score_all`.
    """

    name: ClassVar[str]
    config_cls: ClassVar[type[ModelConfig]]
    config: ModelConfig

    @property
    def method(self) -> str:
        """Name used in result files: the model name, plus `_<variant>` if set."""
        if self.config.variant:
            return f"{self.name}_{self.config.variant}"
        return self.name

    @abstractmethod
    def fit(self, split: Split) -> None:
        """Train on `split`.

        May read `split.train()` and `split.valid()`, never `split.test()`.
        """

    @abstractmethod
    def score_all(
        self, users: Sequence[int], histories: Sequence[History]
    ) -> torch.Tensor:
        """Score every item for every case.

        Args:
            users: User index of each case.
            histories: Input history of each case, oldest first.

        Returns:
            A `(len(users), n_items)` tensor; row `r` belongs to case `r`.
        """

    def score_candidates(
        self,
        users: Sequence[int],
        histories: Sequence[History],
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Scores of the candidate items only, for the sampled evaluation.

        Args:
            users: User index of each case.
            histories: Input history of each case.
            candidates: `(len(users), c)` item indices; column 0 is the answer.

        Returns:
            A `(len(users), c)` tensor of scores, in the order of `candidates`.
        """
        with torch.no_grad():
            scores = self._checked_scores(users, histories)
            return scores.gather(1, candidates.to(scores.device)).cpu()

    def recommend(
        self, users: Sequence[int], histories: Sequence[History], k: int
    ) -> list[list[int]]:
        """Top `k` items per case, best first, leaving out items in the history.

        Raises:
            ValueError: If `k` is out of range or a history leaves fewer than
                `k` items to recommend.
        """
        with torch.no_grad():
            # A float copy: the model may still hold the tensor it returned,
            # and -inf needs a floating-point type.
            scores = self._checked_scores(users, histories).to(torch.float32, copy=True)
            n_items = scores.shape[1]
            if not 1 <= k <= n_items:
                raise ValueError(f"k must be between 1 and {n_items}, got {k}")
            rows = [row for row, history in enumerate(histories) for _ in history]
            cols = [item for history in histories for item in history]
            scores[rows, cols] = float("-inf")
            left = torch.isfinite(scores).sum(dim=1)
            if len(users) and int(left.min()) < k:
                raise ValueError(f"a history leaves fewer than k={k} items")
            top: list[list[int]] = scores.topk(k, dim=1).indices.tolist()
            return top

    def _checked_scores(
        self, users: Sequence[int], histories: Sequence[History]
    ) -> torch.Tensor:
        if len(users) != len(histories):
            raise ValueError(f"{len(users)} users for {len(histories)} histories")
        scores = self.score_all(users, histories)
        if scores.dim() != 2 or scores.shape[0] != len(users):
            raise ValueError(
                f"{type(self).__name__}.score_all returned shape "
                f"{tuple(scores.shape)}, expected ({len(users)}, n_items)"
            )
        return scores
