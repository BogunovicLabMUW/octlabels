"""Metadata keys with a defined meaning, shared by all annotation kinds."""

from __future__ import annotations

import math
import numbers
from typing import Any, Callable, Dict, NamedTuple


class InfoKey(NamedTuple):
    """A documented ``info`` key: its meaning and a check for its value."""

    description: str
    is_valid: Callable[[Any], bool]


def _is_integer(value: Any) -> bool:
    # bool is a subclass of int, but True is not a valid size or index.
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_positive_integer(value: Any) -> bool:
    return _is_integer(value) and value > 0


def _is_non_negative_integer(value: Any) -> bool:
    return _is_integer(value) and value >= 0


def _is_positive_number(value: Any) -> bool:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    return math.isfinite(value) and value > 0


def _is_text(value: Any) -> bool:
    return isinstance(value, str)


def _is_laterality(value: Any) -> bool:
    return value in ("OD", "OS")


def _is_source(value: Any) -> bool:
    return value in ("manual", "model", "corrected")


INFO_KEYS: Dict[str, InfoKey] = {
    "width": InfoKey("Number of A-scans.", _is_positive_integer),
    "height": InfoKey("Pixels per A-scan.", _is_positive_integer),
    "idx": InfoKey("Index of the B-scan in its volume.", _is_non_negative_integer),
    "axial_resolution_um": InfoKey(
        "Micrometres per pixel along the A-scan.", _is_positive_number
    ),
    "lateral_resolution_um": InfoKey(
        "Micrometres between neighbouring A-scans.", _is_positive_number
    ),
    "bscan_spacing_um": InfoKey(
        "Micrometres between neighbouring B-scans.", _is_positive_number
    ),
    "laterality": InfoKey("Eye: 'OD' (right) or 'OS' (left).", _is_laterality),
    "subject_id": InfoKey("Pseudonymous subject identifier.", _is_text),
    "scan_id": InfoKey("Identifier of the scan.", _is_text),
    "device": InfoKey("Acquisition device, e.g. 'Spectralis'.", _is_text),
    "source": InfoKey(
        "Origin of the annotations: 'manual', 'model' or 'corrected'.", _is_source
    ),
    "grader": InfoKey("Who annotated the B-scan.", _is_text),
    "model": InfoKey("Model name/version that produced the annotations.", _is_text),
    "format_version": InfoKey("Version of the file format.", _is_positive_integer),
}
"""``info`` keys with a defined meaning. All are optional; ``validate()`` checks the
ones present, and transforms keep the resolutions correct. Any other key is allowed
and passed through unchanged. Do not store direct patient identifiers: files are
meant to be shared.
"""
