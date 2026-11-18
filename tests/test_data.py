import pytest

from sasrec2023.data import Split

ITEMS = tuple(f"i{j}" for j in range(6))


def two_users() -> Split:
    return Split(((0, 1, 2, 3), (4, 5, 0)), ("ua", "ub"), ITEMS)


def test_counts():
    split = two_users()
    assert (split.n_users, split.n_items, split.n_interactions) == (2, 6, 7)


def test_leave_one_out():
    split = two_users()
    assert split.train() == [(0, 1), (4,)]
    assert split.valid() == ([(0, 1), (4,)], [2, 5])
    assert split.test() == ([(0, 1, 2), (4, 5)], [3, 0])


def test_test_input_is_validation_input_plus_validation_answer():
    split = two_users()
    (valid_in, valid_out), (test_in, _) = split.valid(), split.test()
    assert all(
        t == v + (a,) for t, v, a in zip(test_in, valid_in, valid_out, strict=True)
    )


def test_rejects_a_user_with_two_items():
    with pytest.raises(ValueError, match="1 users have fewer than 3 items"):
        Split(((0, 1, 2), (3, 4)), ("ua", "ub"), ITEMS)


def test_rejects_unknown_item_index():
    with pytest.raises(ValueError, match="outside 0..5"):
        Split(((0, 1, 6),), ("ua",), ITEMS)


def test_rejects_mismatched_users():
    with pytest.raises(ValueError, match="2 sequences for 1 users"):
        Split(((0, 1, 2), (3, 4, 5)), ("ua",), ITEMS)
