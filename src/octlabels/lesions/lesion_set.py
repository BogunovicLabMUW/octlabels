"""Lesion masks of one B-scan."""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from octlabels.groups import NamedGroup
from octlabels.lesions.packing import (
    DEFAULT_MAIN_BITS,
    FileLike,
    load_lesions,
    names_or_default,
    save_lesions,
)


class Lesion:
    """A binary lesion mask of one B-scan.

    Args:
        name: Lesion name, e.g. ``"SRF"``.
        mask: 2D mask; nonzero means present. Stored as uint8 0/1.
        main: Whether this is a main lesion: main lesions are mutually
            exclusive (e.g. fluid compartments); all others are overlays that
            may overlap anything (e.g. HRF). Decides how it is saved.
    """

    def __init__(self, name: str, mask: ArrayLike, main: bool = False) -> None:
        self.name = name
        self.main = main
        self._mask = self._to_mask(mask)

    def _to_mask(self, mask: ArrayLike) -> np.ndarray:
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError(
                f"Lesion '{self.name}': mask must be 2D, got {array.shape}."
            )
        return (array > 0).astype(np.uint8)

    @property
    def mask(self) -> np.ndarray:
        """The mask, uint8 0/1. Shape: [height, width]."""
        return self._mask

    @mask.setter
    def mask(self, mask: ArrayLike) -> None:
        array = self._to_mask(mask)
        if array.shape != self._mask.shape:
            raise ValueError(
                f"Lesion '{self.name}': new mask has shape {array.shape}, "
                f"expected {self._mask.shape}."
            )
        self._mask = array

    @property
    def shape(self) -> Tuple[int, int]:
        """``(height, width)``."""
        height, width = self._mask.shape
        return height, width

    def copy(self) -> Lesion:
        """Returns a deep copy."""
        return Lesion(self.name, self._mask.copy(), self.main)

    def __repr__(self) -> str:
        kind = "main" if self.main else "overlay"
        return f"Lesion(name={self.name!r}, {kind}, shape={self.shape})"


class LesionSet:
    """Lesion masks of one B-scan.

    Attributes:
        info: Metadata such as ``height``, ``width`` and ``idx``. Only the
            masks and their names are stored in the PNG file.
        lesions: The masks by name; behaves like a dict.
    """

    def __init__(self, info: Optional[Mapping[str, Any]] = None) -> None:
        self.info: Dict[str, Any] = dict(info or {})
        self.lesions: NamedGroup[Lesion] = NamedGroup(Lesion, self._claim_shape)

    def _claim_shape(self, lesion: Lesion) -> None:
        height, width = lesion.shape
        self._claim_size(lesion.name, "height", height)
        self._claim_size(lesion.name, "width", width)

    def _claim_size(self, lesion_name: str, key: str, size: int) -> None:
        if key not in self.info:
            self.info[key] = size
        elif self.info[key] != size:
            raise ValueError(
                f"Lesion '{lesion_name}' has {key} {size}, "
                f"LesionSet has {self.info[key]}."
            )

    @property
    def height(self) -> Optional[int]:
        """Rows per B-scan, from ``info["height"]``."""
        return self.info.get("height")

    @property
    def width(self) -> Optional[int]:
        """A-scans per B-scan, from ``info["width"]``."""
        return self.info.get("width")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """``(num_lesions, height, width)``, missing dimensions are 0."""
        return (len(self.lesions), self.height or 0, self.width or 0)

    @property
    def main_lesion_names(self) -> List[str]:
        """Names of the main (mutually exclusive) lesions, in order."""
        return [name for name, lesion in self.lesions.items() if lesion.main]

    @main_lesion_names.setter
    def main_lesion_names(self, names: Sequence[str]) -> None:
        unknown = [name for name in names if name not in self.lesions]
        if unknown:
            raise KeyError(f"Unknown lesions: {unknown}.")
        for name, lesion in self.lesions.items():
            lesion.main = name in names

    def __len__(self) -> int:
        """Number of lesions."""
        return len(self.lesions)

    def __contains__(self, name: str) -> bool:
        return name in self.lesions

    def __getitem__(self, key: Union[int, str]) -> Lesion:
        """Returns a lesion by position or name."""
        if isinstance(key, numbers.Integral):
            return list(self.lesions.values())[key]
        if isinstance(key, str):
            return self.lesions[key]
        raise TypeError(f"Invalid key type for LesionSet: {type(key)}.")

    def __repr__(self) -> str:
        return f"LesionSet(info={self.info}, lesions={list(self.lesions)})"

    def add_lesion(self, lesion: Lesion) -> None:
        """Adds a lesion. Raises ValueError for duplicates or a wrong shape."""
        self.lesions.add(lesion)

    def add_lesion_raw(self, name: str, mask: ArrayLike, main: bool = False) -> None:
        """Creates and adds a lesion from a mask."""
        self.lesions.add(Lesion(name, mask, main))

    def copy(self) -> LesionSet:
        """Returns a deep copy."""
        return self._select(list(self.lesions), create_missing=False)

    def _select(self, names: Sequence[str], create_missing: bool) -> LesionSet:
        result = LesionSet(self.info)
        for name in names:
            if name in self.lesions:
                result.lesions.add(self.lesions[name].copy())
            elif create_missing:
                shape = (self.height or 0, self.width or 0)
                result.lesions.add(Lesion(name, np.zeros(shape)))
            else:
                raise KeyError(f"Lesion '{name}' not found.")
        return result

    def subset(self, names: Sequence[str]) -> LesionSet:
        """Returns a copy with only the given lesions, in this order.
        Raises KeyError for unknown names."""
        return self._select(names, create_missing=False)

    def ensure(self, names: Sequence[str]) -> LesionSet:
        """Returns a copy with exactly the given lesions, in this order;
        missing ones are empty overlays."""
        return self._select(names, create_missing=True)

    def to_array(self, names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Stacks the masks. Shape: [num_lesions, height, width], uint8.

        Args:
            names: Lesions to return, in this order; missing ones are empty
                masks. Defaults to all lesions.
        """
        if names is None:
            lesion_set = self
        else:
            lesion_set = self.ensure(names)
        if not lesion_set.lesions:
            return np.zeros((0, self.height or 0, self.width or 0), dtype=np.uint8)
        return np.stack([lesion.mask for lesion in lesion_set.lesions.values()])

    @classmethod
    def from_array(
        cls,
        masks: ArrayLike,
        names: Optional[Sequence[str]] = None,
        info: Optional[Mapping[str, Any]] = None,
        main_lesion_names: Sequence[str] = (),
    ) -> LesionSet:
        """Creates a LesionSet from stacked masks.

        Args:
            masks: Masks, nonzero = present. Shape: [num_lesions, height, width].
            names: One name per mask. Defaults to ``"Lesion 0"``, ...
            info: Further metadata.
            main_lesion_names: Which of ``names`` are main lesions.
        """
        masks = np.asarray(masks)
        if masks.ndim != 3:
            raise ValueError(f"masks must be (C, H, W), got shape {masks.shape}.")
        if names is None:
            names = [f"Lesion {i}" for i in range(masks.shape[0])]
        if len(names) != masks.shape[0]:
            raise ValueError(f"Got {len(names)} names for {masks.shape[0]} masks.")
        lesion_set = cls(info)
        for name, mask in zip(names, masks, strict=True):
            lesion_set.add_lesion_raw(name, mask, main=name in main_lesion_names)
        return lesion_set

    def save(
        self,
        path: FileLike,
        main_lesion_names: Optional[Sequence[str]] = None,
        main_lesion_bit_count: int = DEFAULT_MAIN_BITS,
    ) -> None:
        """Saves to a PNG, see [`save_lesions`][].

        Args:
            path: Output file (parent folders are created), or an open binary
                file.
            main_lesion_names: Overrides which lesions are main lesions;
                defaults to each lesion's ``main`` flag.
            main_lesion_bit_count: Bits for main lesions.
        """
        if isinstance(path, (str, Path)):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        if main_lesion_names is None:
            main_lesion_names = self.main_lesion_names
        main = [name for name in self.lesions if name in main_lesion_names]
        overlays = [name for name in self.lesions if name not in main_lesion_names]

        # to_array keeps the image size even for an empty list of names.
        save_lesions(
            path,
            self.to_array(main),
            self.to_array(overlays),
            main,
            overlays,
            main_lesion_bit_count,
        )

    @classmethod
    def load(
        cls, path: FileLike, main_lesion_bit_count: Optional[int] = None
    ) -> LesionSet:
        """Loads a PNG written by [`save`][]. Main lesions come first.

        Args:
            path: The PNG file.
            main_lesion_bit_count: Defaults to the value stored in the file.
        """
        main, overlays, main_names, overlay_names = load_lesions(
            path, main_lesion_bit_count
        )
        main_names = names_or_default(main_names, len(main), "L")
        overlay_names = names_or_default(overlay_names, len(overlays), "Ov")

        _, height, width = main.shape
        lesion_set = cls({"height": height, "width": width})
        for name, mask in zip(main_names, main, strict=True):
            lesion_set.add_lesion_raw(name, mask, main=True)
        for name, mask in zip(overlay_names, overlays, strict=True):
            lesion_set.add_lesion_raw(name, mask)
        return lesion_set
