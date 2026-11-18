from itertools import pairwise

from sasrec2023.constants import HISTORY_GROUPS, N_NEGATIVES, TRAIN_SEEDS


def test_history_groups_cover_every_length_once():
    assert HISTORY_GROUPS[0][0] == 1
    assert HISTORY_GROUPS[-1][1] is None
    for (_, high), (low, _) in pairwise(HISTORY_GROUPS):
        assert high is not None
        assert low == high + 1


def test_training_seeds_are_distinct():
    assert len(set(TRAIN_SEEDS)) == len(TRAIN_SEEDS) == 3


def test_one_answer_and_the_negatives_make_101_candidates():
    assert 1 + N_NEGATIVES == 101
