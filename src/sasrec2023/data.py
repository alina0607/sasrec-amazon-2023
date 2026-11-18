"""The data every model and script reads: one `Split` per category.

`Split` is the contract between the data loader and everything else. Models
only see integer indices: user `u` is position `u` of `Split.sequences`, and
item `j` is column `j` of every score matrix.
"""

from dataclasses import dataclass

# One user's items, oldest first, as item indices.
History = tuple[int, ...]


@dataclass(frozen=True)
class Split:
    """Every user's item sequence, cut by leave-one-out (paper Section IV-A).

    Each sequence holds the user's training items, then the validation item,
    then the test item, oldest first.

    Attributes:
        sequences: Item indices per user.
        users: Original `user_id` of each user index.
        items: Original `parent_asin` of each item index.

    Raises:
        ValueError: If the lengths do not match, a user has fewer than 3
            items, or a sequence holds an index outside `range(n_items)`.
    """

    sequences: tuple[History, ...]
    users: tuple[str, ...]
    items: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.sequences) != len(self.users):
            raise ValueError(
                f"{len(self.sequences)} sequences for {len(self.users)} users"
            )
        short = sum(len(seq) < 3 for seq in self.sequences)
        if short:
            raise ValueError(f"{short} users have fewer than 3 items")
        bad = {i for seq in self.sequences for i in seq if not 0 <= i < self.n_items}
        if bad:
            raise ValueError(
                f"item indices outside 0..{self.n_items - 1}: {sorted(bad)[:5]}"
            )

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_interactions(self) -> int:
        return sum(map(len, self.sequences))

    def train(self) -> list[History]:
        """Training sequences: every item but the last two."""
        return [seq[:-2] for seq in self.sequences]

    def valid(self) -> tuple[list[History], list[int]]:
        """Validation cases: (training sequence, second-to-last item) per user."""
        return [seq[:-2] for seq in self.sequences], [seq[-2] for seq in self.sequences]

    def test(self) -> tuple[list[History], list[int]]:
        """Test cases: (training sequence + validation item, last item) per user.

        The input includes the validation item, as in the paper (Section IV-A).
        """
        return [seq[:-1] for seq in self.sequences], [seq[-1] for seq in self.sequences]
