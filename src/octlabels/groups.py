"""A name-keyed collection of annotations, shared by all annotation kinds."""

from __future__ import annotations

from typing import (
    Callable,
    Dict,
    Generic,
    Iterator,
    Mapping,
    MutableMapping,
    Protocol,
    Type,
    TypeVar,
)


class Named(Protocol):
    """Anything with a ``name``."""

    name: str


N = TypeVar("N", bound=Named)


class NamedGroup(MutableMapping[str, N], Generic[N]):
    """Ordered, name-keyed collection of one kind of annotation.

    Behaves like a dict whose keys are the members' names.

    Args:
        kind: The class of the members.
        check: Called with every member before it is stored, e.g. to enforce
            the size of the owning B-scan. Raises to reject the member.
    """

    def __init__(self, kind: Type[N], check: Callable[[N], None]) -> None:
        self.kind = kind
        self._check = check
        self._items: Dict[str, N] = {}

    def __getitem__(self, name: str) -> N:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"{self.kind.__name__} '{name}' not found.") from None

    def __setitem__(self, name: str, member: N) -> None:
        if not isinstance(member, self.kind):
            raise TypeError(f"Expected {self.kind.__name__}, got {type(member)}.")
        if member.name != name:
            raise ValueError(f"Key '{name}' differs from name '{member.name}'.")
        self._check(member)
        self._items[name] = member

    def __delitem__(self, name: str) -> None:
        del self._items[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.kind.__name__}, {list(self._items)})"

    def add(self, member: N) -> None:
        """Adds a member. Raises ValueError if the name already exists."""
        if member.name in self._items:
            raise ValueError(f"{self.kind.__name__} '{member.name}' already exists.")
        self[member.name] = member

    def rename(self, mapping: Mapping[str, str]) -> None:
        """Renames members in place, keeping their order.

        Raises:
            ValueError: If two members would end up with the same name.
        """
        new_names = [mapping.get(name, name) for name in self._items]
        if len(set(new_names)) != len(new_names):
            raise ValueError(f"Renaming would create duplicate names: {new_names}.")
        members = list(self._items.values())
        self._items = {}
        for member, name in zip(members, new_names, strict=True):
            member.name = name
            self._items[name] = member
