"""Metaclass that checks every model class and registers it by name.

A model class must declare `name` and `config_cls`, and its `__init__` must
set `self.config`. The metaclass checks the first two when Python reads the
`class` statement, and the third right after the object is built, so a
mistake fails at once with a message instead of halfway through a run.

Each model is registered under its `name`; `scripts/run.py --model bpr` finds
the class with `RecommenderMeta.get("bpr")`.
"""

from abc import ABCMeta
from types import MappingProxyType
from typing import Any, ClassVar

from sasrec2023.config import ModelConfig


class RecommenderMeta(ABCMeta):
    """Metaclass of `ScoringRecommender` and so of every model.

    It extends `ABCMeta` because the base class has abstract methods, and a
    class can have only one metaclass.
    """

    _registry: ClassVar[dict[str, type]] = {}

    name: str
    config_cls: type[ModelConfig]

    def __new__(
        mcls,
        cls_name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwargs: Any,
    ) -> type:
        cls = super().__new__(mcls, cls_name, bases, namespace, **kwargs)
        # Classes with abstract methods are bases, not models.
        if cls.__abstractmethods__:
            return cls
        # `name` must be set on the class itself: a subclass that inherited
        # it would replace its parent in the registry.
        name = namespace.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError(f"{cls_name} must set a class attribute 'name'")
        config_cls = getattr(cls, "config_cls", None)
        if not (isinstance(config_cls, type) and issubclass(config_cls, ModelConfig)):
            raise TypeError(
                f"{cls_name} must set 'config_cls' to a ModelConfig subclass"
            )
        if name in mcls._registry:
            other = mcls._registry[name]
            raise TypeError(
                f"{cls_name}: name {name!r} is already used by {other.__qualname__}"
            )
        mcls._registry[name] = cls
        return cls

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # Runs after every `__init__` in the hierarchy has finished.
        model = super().__call__(*args, **kwargs)
        if not isinstance(getattr(model, "config", None), cls.config_cls):
            raise TypeError(
                f"{cls.__qualname__}.__init__ must set self.config to a "
                f"{cls.config_cls.__qualname__}"
            )
        return model

    @classmethod
    def get(cls, name: str) -> type:
        """Return the model class registered as `name`.

        Raises:
            KeyError: If no model has that name; the message lists the names.
        """
        try:
            return cls._registry[name]
        except KeyError:
            known = ", ".join(sorted(cls._registry)) or "none"
            raise KeyError(f"no model named {name!r}; registered: {known}") from None

    @classmethod
    def registry(cls) -> MappingProxyType[str, type]:
        """Read-only view of the registry: model name -> class."""
        return MappingProxyType(cls._registry)
