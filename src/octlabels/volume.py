"""A stack of per-B-scan annotation sets, stored as one file per B-scan."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np


class BScanSet(Protocol):
    """What a volume needs from its per-B-scan sets."""

    info: Dict[str, Any]

    def copy(self) -> Any: ...


S = TypeVar("S", bound=BScanSet)
V = TypeVar("V", bound="Volume[Any]")

ZIP_COMPRESSION: Dict[str, int] = {
    "deflate": zipfile.ZIP_DEFLATED,
    "lzma": zipfile.ZIP_LZMA,
    "none": zipfile.ZIP_STORED,
}
"""Compression methods for volume zip files."""


class Volume(Generic[S]):
    """Ordered list of per-B-scan sets, saved as ``NNN.<suffix>`` files.

    A volume is stored as a folder of B-scan files or as a zip file of them;
    either way, single B-scans can be read without the others. Subclasses set
    ``FILE_PATTERN``, a regex whose first group is the B-scan number, and
    implement ``save``/``load`` with ``_save_sets`` and ``_read_sets``.

    Args:
        sets: Initial B-scans.
    """

    FILE_PATTERN: ClassVar[re.Pattern[str]]

    def __init__(self, sets: Optional[Sequence[S]] = None) -> None:
        self.sets: List[S] = list(sets or [])

    @property
    def info(self) -> Dict[str, Any]:
        """Metadata of the first B-scan (a copy), or ``{}`` if empty."""
        if not self.sets:
            return {}
        return dict(self.sets[0].info)

    def __len__(self) -> int:
        return len(self.sets)

    def __iter__(self) -> Iterator[S]:
        return iter(self.sets)

    def __getitem__(self, key: Union[int, slice, Sequence[int]]) -> Any:
        """Returns a B-scan for an int, or a volume of the same type (sharing
        the B-scans) for a slice or a list of indices."""
        if isinstance(key, slice):
            return type(self)(self.sets[key])
        if isinstance(key, (list, tuple, np.ndarray)):
            return type(self)([self.sets[i] for i in key])
        return self.sets[key]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} B-scans, info={self.info})"

    def copy(self: V) -> V:
        """Returns a deep copy."""
        return type(self)([bscan.copy() for bscan in self.sets])

    def _common_names(
        self, names: Optional[Sequence[str]], members: Callable[[S], Iterable[str]]
    ) -> List[str]:
        """Returns ``names``, or else the members of the B-scans, which must
        then be the same in every B-scan."""
        if names is not None:
            return list(names)
        if not self.sets:
            return []
        first = list(members(self.sets[0]))
        for bscan in self.sets:
            current = list(members(bscan))
            if current != first:
                raise ValueError(
                    f"B-scan {bscan.info.get('idx')} has {current}, the first has "
                    f"{first}. Pass names to select (missing ones are zero-filled)."
                )
        return first

    @staticmethod
    def _stack(arrays: List[np.ndarray]) -> np.ndarray:
        """Stacks per-B-scan arrays, with a clear error if their shapes differ."""
        shapes = sorted({array.shape for array in arrays})
        if len(shapes) > 1:
            raise ValueError(f"B-scans have different shapes: {shapes}.")
        return np.stack(arrays)

    def _save_sets(
        self,
        target: Union[str, Path],
        suffix: str,
        to_bytes: Callable[[S], bytes],
        zip_compression: Optional[str] = None,
    ) -> None:
        """Saves every B-scan as ``NNN<suffix>``, named after ``info["idx"]``
        (or the position) with three digits, e.g. ``007.yml``.

        Args:
            target: A folder, or a path ending in ``.zip`` for a zip file.
            suffix: File name ending of one B-scan, e.g. ``".yml.gz"``.
            to_bytes: Returns the file contents of one B-scan.
            zip_compression: For zip files: ``"deflate"`` (default), ``"lzma"``
                or ``"none"``.

        Raises:
            ValueError: If two B-scans have the same ``idx``, or for an
                unknown ``zip_compression``.
        """
        files: Dict[str, S] = {}
        for position, bscan in enumerate(self.sets):
            idx = int(bscan.info.get("idx", position))
            file_name = f"{idx:03d}{suffix}"
            if file_name in files:
                raise ValueError(f"Two B-scans would be saved as '{file_name}'.")
            files[file_name] = bscan

        target = Path(target)
        if not is_zip_path(target):
            target.mkdir(parents=True, exist_ok=True)
            for file_name, bscan in files.items():
                (target / file_name).write_bytes(to_bytes(bscan))
            return

        zip_compression = zip_compression or "deflate"
        if zip_compression not in ZIP_COMPRESSION:
            raise ValueError(
                f"Unknown zip compression {zip_compression!r}, "
                f"use {list(ZIP_COMPRESSION)}."
            )
        method = ZIP_COMPRESSION[zip_compression]
        level = 9 if method == zipfile.ZIP_DEFLATED else None
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", method, compresslevel=level) as archive:
            for file_name, bscan in files.items():
                archive.writestr(file_name, to_bytes(bscan))

    @classmethod
    def _read_sets(
        cls,
        source: Union[str, Path],
        from_bytes: Callable[[bytes, str], S],
        only: Optional[int] = None,
    ) -> List[S]:
        """Reads the B-scans of a folder or zip file, in order of their number.

        Args:
            source: A folder or a zip file.
            from_bytes: Builds one B-scan from its file contents and file name.
            only: Read only the B-scan with this number.

        Raises:
            NotADirectoryError: If ``source`` is neither a folder nor a zip file.
            FileNotFoundError: If it contains no B-scan files.
            KeyError: If ``only`` is given and there is no such B-scan.
            ValueError: If a B-scan number occurs twice.
        """
        source = Path(source)
        if source.is_dir():
            names = [path.name for path in source.iterdir()]
            entries = cls._select_entries(source, names, only)
            return [
                cls._build(from_bytes, (source / name).read_bytes(), name, idx)
                for idx, name in entries
            ]

        if not (source.is_file() and zipfile.is_zipfile(source)):
            raise NotADirectoryError(f"'{source}' is neither a folder nor a zip file.")
        with zipfile.ZipFile(source) as archive:
            entries = cls._select_entries(source, archive.namelist(), only)
            return [
                cls._build(from_bytes, archive.read(name), name, idx)
                for idx, name in entries
            ]

    @classmethod
    def _select_entries(
        cls, source: Path, names: Iterable[str], only: Optional[int]
    ) -> List[Tuple[int, str]]:
        """Returns ``(number, name)`` of the B-scan files among ``names``,
        ordered by number. Files in zip subfolders count too."""
        found: Dict[int, str] = {}
        for name in names:
            match = cls.FILE_PATTERN.match(PurePosixPath(name).name)
            if match is None:
                continue
            idx = int(match.group(1))
            if idx in found:
                raise ValueError(
                    f"B-scan {idx} is in '{source}' twice: '{found[idx]}' and '{name}'."
                )
            found[idx] = name

        if not found:
            raise FileNotFoundError(
                f"No files matching {cls.FILE_PATTERN.pattern} in '{source}'."
            )
        if only is None:
            return sorted(found.items())
        if only not in found:
            raise KeyError(f"B-scan {only} not found in '{source}'.")
        return [(only, found[only])]

    @staticmethod
    def _build(
        from_bytes: Callable[[bytes, str], S], data: bytes, name: str, idx: int
    ) -> S:
        """Builds a B-scan; ``info["idx"]`` falls back to the file's number."""
        bscan = from_bytes(data, name)
        bscan.info.setdefault("idx", idx)
        return bscan


def is_zip_path(path: Union[str, Path]) -> bool:
    """Whether ``path`` names a zip file (by its ``.zip`` ending)."""
    return Path(path).suffix.lower() == ".zip"
