from abc import ABC, abstractmethod
from dataclasses import dataclass

import pytest

from sasrec2023.config import ModelConfig
from sasrec2023.registry import RecommenderMeta


class Base(ABC, metaclass=RecommenderMeta):
    """Stands in for ScoringRecommender: abstract, so it is not registered."""

    @abstractmethod
    def fit(self) -> None: ...


@dataclass(frozen=True)
class ToyConfig(ModelConfig):
    width: int = 3


class Toy(Base):
    name = "registry_toy"
    config_cls = ToyConfig

    def __init__(self, config: ToyConfig | None = None) -> None:
        self.config = config or ToyConfig()

    def fit(self) -> None:
        pass


def test_a_model_is_registered_by_name_and_bases_are_not():
    assert RecommenderMeta.get("registry_toy") is Toy
    assert Base not in RecommenderMeta.registry().values()


def test_unknown_name_lists_the_known_ones():
    with pytest.raises(KeyError, match="registry_toy"):
        RecommenderMeta.get("no_such_model")


def test_config_is_frozen():
    with pytest.raises(AttributeError):
        Toy().config.width = 4  # type: ignore[misc]


def test_missing_name_fails_at_the_class_statement():
    with pytest.raises(TypeError, match="must set a class attribute 'name'"):

        class NoName(Base):
            config_cls = ToyConfig

            def fit(self) -> None:
                pass


def test_a_subclass_must_set_its_own_name():
    with pytest.raises(TypeError, match="must set a class attribute 'name'"):

        class Child(Toy):
            pass


def test_config_cls_must_be_a_model_config():
    with pytest.raises(TypeError, match="ModelConfig subclass"):

        class BadConfig(Base):
            name = "registry_bad_config"
            config_cls = dict  # type: ignore[assignment]

            def fit(self) -> None:
                pass


def test_names_are_unique():
    with pytest.raises(TypeError, match="already used by Toy"):

        class Twin(Base):
            name = "registry_toy"
            config_cls = ToyConfig

            def fit(self) -> None:
                pass


def test_init_must_set_the_config():
    class Forgetful(Base):
        name = "registry_forgetful"
        config_cls = ToyConfig

        def fit(self) -> None:
            pass

    with pytest.raises(TypeError, match="must set self.config to a ToyConfig"):
        Forgetful()


def test_config_of_another_type_is_rejected():
    class WrongType(Base):
        name = "registry_wrong_type"
        config_cls = ToyConfig

        def __init__(self) -> None:
            self.config = ModelConfig()

        def fit(self) -> None:
            pass

    with pytest.raises(TypeError, match="ToyConfig"):
        WrongType()
