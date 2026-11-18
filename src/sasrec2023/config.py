"""Model configurations.

A configuration holds every number that changes a model's result. It is a
frozen dataclass: once built it cannot change, so the values recorded for a
run are the values the model used.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Fields every model configuration has. Each model subclasses it.

    Attributes:
        seed: Seed of every random number generator used in `fit`.
        variant: Ablation name. Result files use `<model name>_<variant>`,
            for example `sasrec_nope`; empty for the main model.
    """

    seed: int = 1
    variant: str = ""
