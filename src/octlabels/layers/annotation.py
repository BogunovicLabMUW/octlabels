"""A-scan aligned annotations.

Every annotation stores one ``values`` entry per A-scan of a B-scan, plus any
number of named properties (``confidence``, ``uncertainty`` or custom ones such
as ``snr``) that are aligned with ``values``: they always have the same length.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Callable, ClassVar, Dict, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

_PROPERTY_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

A = TypeVar("A", bound="AScanAnnotation")


class AScanAnnotation(ABC):
    """Abstract base class for data with one entry per A-scan.

    Subclasses define the dtype and validity rules of ``values``. Properties are
    stored by name; ``confidence`` (uint8) and ``uncertainty`` (uint16) have
    fixed dtypes, custom properties keep the dtype they are given.

    Args:
        name: Name of the annotation, e.g. ``"ILM"`` or ``"DRUSEN"``.
        values: One value per A-scan.
        confidence: Optional per-A-scan confidence.
        uncertainty: Optional per-A-scan normalized uncertainty (0-999).
        **properties: Further A-scan aligned properties.

    Raises:
        ValueError: If an array is not 1D, its length differs from ``values``,
            or ``values`` is invalid for the subclass.
    """

    VALUES_DTYPE: ClassVar[DTypeLike]
    PROPERTY_DTYPES: ClassVar[Dict[str, DTypeLike]] = {
        "confidence": np.uint8,
        "uncertainty": np.uint16,
    }
    """Dtypes for properties with a fixed meaning."""
    REQUIRED_PROPERTIES: ClassVar[Tuple[str, ...]] = ()
    """Properties that are filled with ones when not given."""
    INTERPOLATION: ClassVar[Dict[str, str]] = {"confidence": "min"}
    """How each array is resampled along the A-scan axis: "linear", "nearest" or "min"
    (minimum of both neighbours, conservative for validity flags). Arrays not listed
    use "linear" for floats and "nearest" otherwise.
    """
    VALUES_DECIMALS: ClassVar[Optional[int]] = None
    """Decimals used when writing ``values``; ``None`` writes the shortest exact
    representation.
    """

    def __init__(
        self,
        name: str,
        values: Optional[ArrayLike] = None,
        confidence: Optional[ArrayLike] = None,
        uncertainty: Optional[ArrayLike] = None,
        **properties: Optional[ArrayLike],
    ) -> None:
        self.name = name
        self._values = self._to_values(values)
        self._properties: Dict[str, np.ndarray] = {}

        # confidence and uncertainty come first, so files list them first.
        all_properties = {
            "confidence": confidence,
            "uncertainty": uncertainty,
            **properties,
        }
        for key in self.REQUIRED_PROPERTIES:
            if all_properties.get(key) is None:
                all_properties[key] = np.ones(len(self))
        for key, array in all_properties.items():
            if array is not None:
                self.set_property(key, array)

    @abstractmethod
    def _validate_values(self, values: np.ndarray) -> None:
        """Raises ValueError if ``values`` (before dtype conversion) is invalid."""

    def _to_values(self, values: Optional[ArrayLike]) -> np.ndarray:
        if values is None:
            values = []
        raw = np.asarray(values)
        if raw.ndim != 1:
            raise ValueError(f"'{self.name}': values must be 1D, got {raw.shape}.")
        self._validate_values(raw)
        return np.array(raw, dtype=self.VALUES_DTYPE)

    def __len__(self) -> int:
        """Number of A-scans."""
        return self._values.shape[0]

    @property
    def values(self) -> np.ndarray:
        """Per-A-scan values. Shape: [width]."""
        return self._values

    @values.setter
    def values(self, values: ArrayLike) -> None:
        array = self._to_values(values)
        if self._properties and len(array) != len(self):
            raise ValueError(
                f"'{self.name}': new values have length {len(array)}, but the "
                f"properties have length {len(self)}. Create a new annotation "
                "to change the width."
            )
        self._values = array

    @property
    def properties(self) -> Mapping[str, np.ndarray]:
        """Read-only view of all properties."""
        return MappingProxyType(self._properties)

    def set_property(self, key: str, array: ArrayLike) -> None:
        """Adds or replaces an A-scan aligned property.

        Raises:
            ValueError: If the name is invalid or the length does not match.
        """
        if not _PROPERTY_NAME.match(key) or key == "values":
            raise ValueError(f"Invalid property name: {key!r}.")
        dtype = self.PROPERTY_DTYPES.get(key)
        converted = np.array(array, dtype=dtype)
        if converted.ndim != 1 or converted.shape[0] != len(self):
            raise ValueError(
                f"'{self.name}': property '{key}' has shape {converted.shape}, "
                f"expected ({len(self)},)."
            )
        self._properties[key] = converted

    def remove_property(self, key: str) -> None:
        """Removes a property. Raises KeyError if it does not exist."""
        if key in self.REQUIRED_PROPERTIES:
            raise ValueError(f"Property '{key}' is required for {type(self).__name__}.")
        del self._properties[key]

    @property
    def confidence(self) -> Optional[np.ndarray]:
        """Per-A-scan confidence (uint8), or None if not stored."""
        return self._properties.get("confidence")

    @confidence.setter
    def confidence(self, array: ArrayLike) -> None:
        self.set_property("confidence", array)

    @property
    def uncertainty(self) -> Optional[np.ndarray]:
        """Per-A-scan normalized uncertainty (uint16, 0-999), or None."""
        return self._properties.get("uncertainty")

    @uncertainty.setter
    def uncertainty(self, array: ArrayLike) -> None:
        self.set_property("uncertainty", array)

    def interpolation(self, key: str) -> str:
        """Returns the resampling mode for ``"values"`` or a property name."""
        if key in self.INTERPOLATION:
            return self.INTERPOLATION[key]
        array = self._values if key == "values" else self._properties[key]
        return "linear" if np.issubdtype(array.dtype, np.floating) else "nearest"

    def map_arrays(self: A, func: Callable[[str, np.ndarray], np.ndarray]) -> A:
        """Returns a new annotation with ``func(key, array)`` applied to
        ``values`` (key ``"values"``) and every property.

        Use this for operations that must keep all arrays aligned, such as
        flipping, slicing, padding or resampling.
        """
        properties = {key: func(key, array) for key, array in self._properties.items()}
        return type(self)(self.name, func("values", self._values), **properties)

    @classmethod
    def zeros(
        cls: Type[A],
        name: str,
        width: int,
        property_dtypes: Optional[Mapping[str, DTypeLike]] = None,
    ) -> A:
        """Creates an annotation whose values, required properties and the
        properties named in ``property_dtypes`` are all zero."""
        dtypes = {key: cls.PROPERTY_DTYPES.get(key) for key in cls.REQUIRED_PROPERTIES}
        dtypes.update(property_dtypes or {})
        properties = {
            key: np.zeros(width, dtype=dtype) for key, dtype in dtypes.items()
        }
        return cls(name, np.zeros(width), **properties)

    def copy(self: A) -> A:
        """Returns a deep copy."""
        return self.map_arrays(lambda _, array: array)

    def _arrays(self) -> Dict[str, np.ndarray]:
        """``values`` followed by all properties."""
        return {"values": self._values, **self._properties}

    def __repr__(self) -> str:
        arrays = [
            f"{key}=[{array.shape[0]}, {array.dtype}]"
            for key, array in self._arrays().items()
        ]
        return f"{type(self).__name__}(name={self.name!r}, {', '.join(arrays)})"

    def __str__(self) -> str:
        lines = [f"{type(self).__name__}: {self.name} ({len(self)} A-scans)"]
        for key, array in self._arrays().items():
            preview = np.array2string(
                array[:10], precision=2, separator=", ", suppress_small=True
            )
            if len(array) > 10:
                preview += "..."
            lines.append(f"  {key}: {preview}")
        return "\n".join(lines)


class Layer(AScanAnnotation):
    """A retinal layer boundary: ``values`` holds the boundary height (row
    index, float32) at each A-scan. A layer always has a ``confidence``
    property; it defaults to all ones (valid).
    """

    VALUES_DTYPE = np.float32
    REQUIRED_PROPERTIES = ("confidence",)
    VALUES_DECIMALS = 2

    def _validate_values(self, values: np.ndarray) -> None:
        if values.size and not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"Layer '{self.name}': values must be numeric.")


class AScanRegion(AScanAnnotation):
    """Presence of a biomarker (e.g. ``"DRUSEN"``): ``values`` is 0 or 1
    (uint8) at each A-scan. ``confidence`` is optional.
    """

    VALUES_DTYPE = np.uint8
    INTERPOLATION = {**AScanAnnotation.INTERPOLATION, "values": "nearest"}

    def _validate_values(self, values: np.ndarray) -> None:
        if values.size and not np.isin(values, (0, 1)).all():
            raise ValueError(f"Region '{self.name}': values must be 0 or 1.")


def uncertainty_from_std(std: ArrayLike) -> np.ndarray:
    """Converts standard deviations to the normalized uncertainty encoding.

    Values are clamped to [1, 100] and mapped linearly to [0, 999].

    Args:
        std: Standard deviations in pixels, any shape.

    Returns:
        uint16 array of the same shape.
    """
    clamped = np.clip(np.asarray(std, dtype=np.float64), 1.0, 100.0)
    return np.rint((clamped - 1.0) / 99.0 * 999.0).astype(np.uint16)
