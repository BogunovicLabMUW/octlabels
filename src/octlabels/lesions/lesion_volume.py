"""A stack of lesion masks, stored as one PNG per B-scan in a folder or zip file."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from octlabels.lesions.lesion_set import LesionSet
from octlabels.lesions.packing import DEFAULT_MAIN_BITS
from octlabels.volume import Volume, is_zip_path


class LesionVolume(Volume[LesionSet]):
    """An ordered list of [`LesionSet`][] objects, one per B-scan.

    Args:
        lesion_sets: Initial B-scans.
    """

    FILE_PATTERN = re.compile(r"^(\d+)\.png$")

    def __init__(self, lesion_sets: Optional[Sequence[LesionSet]] = None) -> None:
        super().__init__(lesion_sets)

    @property
    def lesion_sets(self) -> List[LesionSet]:
        """The B-scans, in order (the list itself, not a copy)."""
        return self.sets

    @lesion_sets.setter
    def lesion_sets(self, lesion_sets: Sequence[LesionSet]) -> None:
        self.sets = list(lesion_sets)

    def add_lesion_set(self, lesion_set: LesionSet) -> None:
        """Appends a B-scan."""
        self.sets.append(lesion_set)

    def to_array(self, names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Stacks the masks. Shape: [num_bscans, num_lesions, height, width], uint8.

        Args:
            names: Lesions to return, in this order; missing ones are empty
                masks. Defaults to the lesions of the B-scans, which must then
                be the same in every B-scan.

        Raises:
            ValueError: If ``names`` is not given and the B-scans have different
                lesions, or if the B-scans have different sizes.
        """
        names = self._common_names(names, lambda lesion_set: lesion_set.lesions)
        if not self.sets:
            return np.zeros((0, len(names), 0, 0), np.uint8)
        return self._stack([lesion_set.to_array(names) for lesion_set in self.sets])

    @classmethod
    def from_array(
        cls,
        masks: ArrayLike,
        names: Optional[Sequence[str]] = None,
        info: Optional[Mapping[str, Any]] = None,
        main_lesion_names: Sequence[str] = (),
    ) -> LesionVolume:
        """Creates a volume from stacked masks.

        Args:
            masks: Masks, nonzero = present.
                Shape: [num_bscans, num_lesions, height, width].
            names: One name per lesion channel, see [`LesionSet.from_array`][].
            info: Metadata for every B-scan; ``idx`` is set to the B-scan index.
            main_lesion_names: Which of ``names`` are main lesions.
        """
        masks = np.asarray(masks)
        if masks.ndim != 4:
            raise ValueError(f"masks must be (B, C, H, W), got shape {masks.shape}.")
        volume = cls()
        for i in range(masks.shape[0]):
            bscan_info = {**(info or {}), "idx": i}
            volume.add_lesion_set(
                LesionSet.from_array(masks[i], names, bscan_info, main_lesion_names)
            )
        return volume

    def save(
        self,
        target: Union[str, Path],
        main_lesion_names: Optional[Sequence[str]] = None,
        main_lesion_bit_count: int = DEFAULT_MAIN_BITS,
        compress: Optional[str] = None,
    ) -> None:
        """Writes one PNG per B-scan, named after ``info["idx"]`` (or the
        position) with three digits: ``000.png``, ``001.png``, ...

        Args:
            target: A folder (created if missing), or a path ending in
                ``.zip`` to write all B-scan files into one zip file.
            main_lesion_names: Overrides which lesions are main lesions;
                defaults to each lesion's ``main`` flag.
            main_lesion_bit_count: Bits for main lesions.
            compress: For zip files only: ``"deflate"`` (default), ``"lzma"``
                or ``"none"``. PNGs are already compressed, so the choice
                matters little.
        """
        if compress is not None and not is_zip_path(target):
            raise ValueError("compress only applies to zip files.")

        def to_bytes(lesion_set: LesionSet) -> bytes:
            buffer = io.BytesIO()
            lesion_set.save(buffer, main_lesion_names, main_lesion_bit_count)
            return buffer.getvalue()

        self._save_sets(target, ".png", to_bytes, compress)

    @classmethod
    def load(
        cls, source: Union[str, Path], main_lesion_bit_count: Optional[int] = None
    ) -> LesionVolume:
        """Reads all ``NNN.png`` files of a folder or zip file, ordered by
        number; ``info["idx"]`` is taken from the file name.

        Raises:
            NotADirectoryError: If ``source`` is neither a folder nor a zip file.
            FileNotFoundError: If it contains no B-scan files.
            ValueError: If a B-scan number occurs twice.
        """

        def from_bytes(data: bytes, name: str) -> LesionSet:
            return LesionSet.load(io.BytesIO(data), main_lesion_bit_count)

        return cls(cls._read_sets(source, from_bytes))

    @classmethod
    def load_bscan(
        cls,
        source: Union[str, Path],
        idx: int,
        main_lesion_bit_count: Optional[int] = None,
    ) -> LesionSet:
        """Reads only B-scan ``idx`` (the number in its file name) of a folder
        or zip file, without reading the other B-scans.

        Raises:
            KeyError: If there is no such B-scan.
        """

        def from_bytes(data: bytes, name: str) -> LesionSet:
            return LesionSet.load(io.BytesIO(data), main_lesion_bit_count)

        return cls._read_sets(source, from_bytes, only=idx)[0]
