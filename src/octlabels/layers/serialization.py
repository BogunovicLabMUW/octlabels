"""Reading and writing layer files.

A file is YAML with an ``info`` mapping and one section per annotation kind::

    info:
      width: 4
      height: 496

    layers:
      - ILM:
        - values: 120,121.5,123,124
        - confid:   1,    1,  0,  1
        - uncrtn:  12,   40,999, 10

    regions:
      - DRUSEN:
        - values: 0,1,1,0

Each row holds one entry per A-scan. By default rows are written as
comma-separated text (which YAML reads as a string); with ``yaml_arrays=True``
they are written as YAML lists (``[120, 121.5, ...]``). Both are read. Rows of
one annotation are padded so their columns line up. Rows other than
``values``, ``confid`` and ``uncrtn`` are custom properties.
"""

from __future__ import annotations

import gzip
import json
import lzma
import numbers
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import numpy as np
import yaml
from numpy.typing import DTypeLike

from octlabels.layers.annotation import AScanAnnotation
from octlabels.layers.layer_set import LayerSet

# The C loader is about 40 times faster; libyaml is not always installed.
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

# Row keys on disk for properties whose in-memory name is longer.
_DISK_KEYS = {"confidence": "confid", "uncertainty": "uncrtn"}
_MEMORY_KEYS = {disk: memory for memory, disk in _DISK_KEYS.items()}

_INTEGER = re.compile(r"^\s*[+-]?\d+\s*$")
# How YAML spells the special float values.
_YAML_SPECIAL_FLOATS = {"nan": ".nan", "inf": ".inf", "-inf": "-.inf"}
# Compression methods and the file name ending each one uses.
_COMPRESSION_SUFFIXES = {"none": "", "gzip": ".gz", "lzma": ".xz"}


# --- Reading -----------------------------------------------------------------


def loads(text: str) -> LayerSet:
    """Parses a LayerSet from text.

    Raises:
        yaml.YAMLError: If the text is not valid YAML.
        ValueError: If the structure or a row is invalid.
    """
    data = yaml.load(text, Loader=_LOADER) or {}
    if not isinstance(data, dict):
        raise ValueError("Expected a mapping with 'info' and annotation sections.")
    for key in data:
        if key != "info" and key not in LayerSet.SECTIONS:
            warnings.warn(f"Ignoring unknown section '{key}'.", stacklevel=2)

    layer_set = LayerSet(data.get("info") or {})
    for section, group in layer_set.sections().items():
        for name, rows in _entries(data.get(section) or []):
            group.add(_build(group.kind, name, rows))
    return layer_set


def _entries(section: List[Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yields ``(name, {row_key: row})`` for each ``- NAME: [- key: row, ...]``."""
    for entry in section:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Expected '- NAME:' entry, got {entry!r:.80}.")
        name, fields = next(iter(entry.items()))
        if fields is None:
            fields = []
        if not isinstance(fields, list):
            raise ValueError(f"'{name}': expected a list of rows, got {fields!r:.80}.")

        rows: Dict[str, Any] = {}
        for field in fields:
            if not isinstance(field, dict):
                raise ValueError(f"'{name}': expected '- key: row', got {field!r:.80}.")
            rows.update(field)
        yield str(name), rows


def _build(
    kind: Type[AScanAnnotation], name: str, rows: Dict[str, Any]
) -> AScanAnnotation:
    values = _parse_row(rows.pop("values", None), kind.VALUES_DTYPE)
    properties = {}
    for disk_key, row in rows.items():
        if row is None:  # an empty property row means "not stored"
            continue
        key = _MEMORY_KEYS.get(disk_key, disk_key)
        properties[key] = _parse_row(row, kind.PROPERTY_DTYPES.get(key))
    return kind(name, values, **properties)


def _parse_row(row: Any, dtype: Optional[DTypeLike]) -> np.ndarray:
    """Converts a row to an array.

    A row is comma-separated text, a YAML list, a single number or empty.
    Without a ``dtype``, rows of integers become int64, others float64.
    """
    if row is None:
        items: List[Any] = []
    elif isinstance(row, str):
        items = [token for token in row.split(",") if token.strip()]
    elif isinstance(row, list):
        items = row
    else:
        items = [row]

    if dtype is None:
        if items and all(_is_integer_item(item) for item in items):
            dtype = np.int64
        else:
            dtype = np.float64
    return np.array(items, dtype=dtype)


def _is_integer_item(item: Any) -> bool:
    if isinstance(item, int):
        return True
    return isinstance(item, str) and _INTEGER.match(item) is not None


def load_bytes(data: bytes, name: str) -> LayerSet:
    """Parses the contents of a ``.yml``, ``.yml.gz`` or ``.yml.xz`` file.

    Args:
        data: The file contents.
        name: File (or zip entry) name; its ending selects the decompression.

    Raises:
        ValueError: If the file contains no annotations.
    """
    if name.endswith(".gz"):
        data = gzip.decompress(data)
    elif name.endswith(".xz"):
        data = lzma.decompress(data)
    layer_set = loads(data.decode("utf-8"))
    if not any(layer_set.sections().values()):
        raise ValueError(f"{name} contains no annotations.")
    return layer_set


def load(path: Union[str, Path]) -> LayerSet:
    """Reads a ``.yml``, ``.yml.gz`` or ``.yml.xz`` file.

    Raises:
        ValueError: If the file contains no annotations.
    """
    path = Path(path)
    return load_bytes(path.read_bytes(), str(path))


# --- Writing -----------------------------------------------------------------


def dumps(layer_set: LayerSet, yaml_arrays: bool = False) -> str:
    """Serializes a LayerSet.

    Args:
        layer_set: The LayerSet to write.
        yaml_arrays: Write rows as YAML lists instead of comma-separated text.
    """
    lines: List[str] = []
    if layer_set.info:
        lines.append("info:")
        for key, value in layer_set.info.items():
            lines.append(f"  {_scalar(key)}: {_scalar(value)}")

    for section, group in layer_set.sections().items():
        if not group:
            continue
        if lines:
            lines.append("")
        lines.append(f"{section}:")
        for annotation in group.values():
            lines.append(f"  - {_scalar(annotation.name)}:")
            lines += _rows(annotation, yaml_arrays)
    return "\n".join(lines) + "\n"


def _rows(annotation: AScanAnnotation, yaml_arrays: bool) -> List[str]:
    """Formats all rows of an annotation, with their columns aligned."""
    arrays = {"values": annotation.values, **annotation.properties}

    cells_by_key = {}
    for key, array in arrays.items():
        decimals = annotation.VALUES_DECIMALS if key == "values" else None
        cells = _format_cells(array, decimals)
        if yaml_arrays:
            cells = [_YAML_SPECIAL_FLOATS.get(cell, cell) for cell in cells]
        cells_by_key[key] = cells

    labels = {key: _DISK_KEYS.get(key, key) + ":" for key in arrays}
    label_width = max(len(label) for label in labels.values())
    cell_width = max(
        (len(cell) for cells in cells_by_key.values() for cell in cells), default=0
    )

    lines = []
    for key, cells in cells_by_key.items():
        padded = [cell.rjust(cell_width) for cell in cells]
        if yaml_arrays:
            row = "[" + ", ".join(padded) + "]"
        else:
            row = ",".join(padded)
        lines.append(f"    - {labels[key].ljust(label_width)} {row}".rstrip())
    return lines


def _format_cells(array: np.ndarray, decimals: Optional[int]) -> List[str]:
    """Formats numbers with as few characters as possible.

    Integers are written as such. Floats are rounded to ``decimals`` places
    (trailing zeros dropped), or written in the shortest exact form if
    ``decimals`` is None.
    """
    if not np.issubdtype(array.dtype, np.floating):
        return [str(value) for value in array.tolist()]
    if decimals is None:
        return [
            np.format_float_positional(value, unique=True, trim="-") for value in array
        ]
    if np.allclose(array, np.round(array), atol=1e-6):
        return [f"{value:.0f}" for value in array]
    return [_drop_trailing_zeros(f"{value:.{decimals}f}") for value in array]


def _drop_trailing_zeros(number: str) -> str:
    """``"1.50"`` -> ``"1.5"``, ``"2.00"`` -> ``"2"``; ``"nan"`` is unchanged."""
    if "." not in number:
        return number
    return number.rstrip("0").rstrip(".")


def _scalar(value: Any) -> str:
    """Formats an info value or a name as a YAML scalar that reads back as the
    same value. Strings are quoted only when needed."""
    if value is None:
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        text = repr(float(value))
        return _YAML_SPECIAL_FLOATS.get(text, text)

    text = str(value)
    if _reads_back_unquoted(text):
        return text
    return json.dumps(text)  # a JSON string is also a valid YAML string


def _reads_back_unquoted(text: str) -> bool:
    """Whether YAML reads ``text`` back as the same string without quotes;
    not the case for e.g. ``"yes"``, ``"1"``, ``"a: b"`` or ``"#3"``."""
    if "\n" in text:
        return False
    try:
        return yaml.load(f"key: {text}", Loader=_LOADER) == {"key": text}
    except yaml.YAMLError:
        return False


def to_bytes(
    layer_set: LayerSet, compress: str = "none", yaml_arrays: bool = False
) -> bytes:
    """Returns the contents of a layer file.

    Args:
        layer_set: The LayerSet to write.
        compress: ``"none"``, ``"gzip"`` or ``"lzma"``.
        yaml_arrays: Write rows as YAML lists, see [`dumps`][].
    """
    compression_suffix(compress)  # raises for an unknown method
    data = dumps(layer_set, yaml_arrays=yaml_arrays).encode("utf-8")
    if compress == "gzip":
        return gzip.compress(data, compresslevel=9)
    if compress == "lzma":
        return lzma.compress(data, preset=9)
    return data


def compression_suffix(compress: str) -> str:
    """The file name ending for a compression method, e.g. ``".gz"``."""
    if compress not in _COMPRESSION_SUFFIXES:
        raise ValueError(
            f"Unknown compression {compress!r}, use {list(_COMPRESSION_SUFFIXES)}."
        )
    return _COMPRESSION_SUFFIXES[compress]


def _compression_from_name(path: Path) -> str:
    for method, suffix in _COMPRESSION_SUFFIXES.items():
        if suffix and path.name.endswith(suffix):
            return method
    return "none"


def save(
    layer_set: LayerSet,
    path: Union[str, Path],
    compress: Optional[str] = None,
    yaml_arrays: bool = False,
) -> Path:
    """Writes a LayerSet to a file, creating parent folders.

    Args:
        layer_set: The LayerSet to write.
        path: Target file. ``.gz`` or ``.xz`` at the end selects compression.
        compress: ``"none"``, ``"gzip"`` or ``"lzma"``; defaults to what the
            file name says. If given, the matching ending is appended when
            missing.
        yaml_arrays: Write rows as YAML lists, see [`dumps`][].

    Returns:
        The path written to.

    Raises:
        ValueError: For an unknown ``compress``, or one that contradicts the
            file name.
    """
    path = Path(path)
    named_compression = _compression_from_name(path)
    if compress is None:
        compress = named_compression
    suffix = compression_suffix(compress)
    if named_compression != "none" and compress != named_compression:
        raise ValueError(
            f"compress={compress!r} contradicts the file name {path.name}."
        )
    if not path.name.endswith(suffix):
        path = path.with_name(path.name + suffix)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(to_bytes(layer_set, compress, yaml_arrays))
    return path
