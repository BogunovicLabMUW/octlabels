"""A stack of B-scans, stored as one file per B-scan in a folder or zip file."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from octlabels.layers import serialization
from octlabels.layers.layer_set import LayerSet
from octlabels.volume import Volume, is_zip_path


class LayerVolume(Volume[LayerSet]):
    """An ordered list of [`LayerSet`][] objects, one per B-scan.

    Args:
        layer_sets: Initial B-scans.
    """

    FILE_PATTERN = re.compile(r"^(\d+)\.yml(\.gz|\.xz)?$")

    def __init__(self, layer_sets: Optional[Sequence[LayerSet]] = None) -> None:
        super().__init__(layer_sets)

    @property
    def layer_sets(self) -> List[LayerSet]:
        """The B-scans, in order (the list itself, not a copy)."""
        return self.sets

    @layer_sets.setter
    def layer_sets(self, layer_sets: Sequence[LayerSet]) -> None:
        self.sets = list(layer_sets)

    def add_layer_set(self, layer_set: LayerSet) -> None:
        """Appends a B-scan."""
        self.sets.append(layer_set)

    def to_array(
        self, names: Optional[Sequence[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns ``(values, confidence)``, each of shape
        [num_bscans, num_layers, width]; NaN values become 0.

        Args:
            names: Layers to return, in this order; missing ones are zero
                (value 0, confidence 0). Defaults to the layers of the B-scans,
                which must then be the same in every B-scan.

        Raises:
            ValueError: If ``names`` is not given and the B-scans have different
                layers, or if the B-scans have different widths.
        """
        names = self._common_names(names, lambda layer_set: layer_set.layers)
        if not self.sets:
            empty_shape = (0, len(names), 0)
            return np.zeros(empty_shape, np.float32), np.zeros(empty_shape, np.uint8)

        values, confidence = [], []
        for layer_set in self.sets:
            bscan_values, bscan_confidence = layer_set.to_array(names)
            values.append(bscan_values)
            confidence.append(bscan_confidence)
        return self._stack(values), self._stack(confidence)

    def values_to_array(self) -> np.ndarray:
        """Layer values. Shape: [num_bscans, num_layers, width]."""
        return np.stack([layer_set.values_to_array() for layer_set in self.sets])

    def confidence_to_array(self) -> np.ndarray:
        """Layer confidences. Shape: [num_bscans, num_layers, width]."""
        return np.stack([layer_set.confidence_to_array() for layer_set in self.sets])

    def uncertainty_to_array(self) -> np.ndarray:
        """Layer uncertainties. Shape: [num_bscans, num_layers, width]."""
        return np.stack([layer_set.uncertainty_to_array() for layer_set in self.sets])

    def values_from_array(self, values: ArrayLike) -> None:
        """Sets layer values. Shape: [num_bscans, num_layers, width]."""
        for layer_set, bscan_values in zip(
            self.sets, self._check_3d(values), strict=True
        ):
            layer_set.values_from_array(bscan_values)

    def confidence_from_array(self, confidence: ArrayLike) -> None:
        """Sets layer confidences. Shape: [num_bscans, num_layers, width]."""
        for layer_set, bscan_confidence in zip(
            self.sets, self._check_3d(confidence), strict=True
        ):
            layer_set.confidence_from_array(bscan_confidence)

    def uncertainty_from_array(self, uncertainty: ArrayLike) -> None:
        """Sets layer uncertainties. Shape: [num_bscans, num_layers, width]."""
        for layer_set, bscan_uncertainty in zip(
            self.sets, self._check_3d(uncertainty), strict=True
        ):
            layer_set.uncertainty_from_array(bscan_uncertainty)

    def _check_3d(self, array: ArrayLike) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim != 3 or array.shape[0] != len(self):
            raise ValueError(
                f"Expected shape ({len(self)}, layers, width), got {array.shape}."
            )
        return array

    @classmethod
    def from_array(
        cls,
        values: ArrayLike,
        confidence: Optional[ArrayLike] = None,
        names: Optional[Sequence[str]] = None,
        info: Optional[Mapping[str, Any]] = None,
        **properties: ArrayLike,
    ) -> LayerVolume:
        """Creates a volume from stacked arrays.

        Args:
            values: Boundary heights. Shape: [num_bscans, num_layers, width].
            confidence: Same shape; defaults to all ones.
            names: Layer names, see [`LayerSet.from_array`][].
            info: Metadata for every B-scan, e.g. ``{"height": 496}``. ``idx``
                is set to the B-scan index.
            **properties: Further properties of the same shape as ``values``.
        """
        values = np.asarray(values)
        if confidence is not None:
            confidence = np.asarray(confidence)
        properties = {key: np.asarray(array) for key, array in properties.items()}

        volume = cls()
        for i in range(values.shape[0]):
            bscan_confidence = None if confidence is None else confidence[i]
            bscan_properties = {key: array[i] for key, array in properties.items()}
            bscan_info = {**(info or {}), "idx": i}
            volume.add_layer_set(
                LayerSet.from_array(
                    values[i], bscan_confidence, names, bscan_info, **bscan_properties
                )
            )
        return volume

    def save(
        self,
        target: Union[str, Path],
        compress: Optional[str] = None,
        yaml_arrays: bool = False,
    ) -> None:
        """Writes one file per B-scan, named after ``info["idx"]`` (or the
        position) with three digits: ``000.yml``, ``001.yml``, ...

        Args:
            target: A folder (created if missing), or a path ending in
                ``.zip`` to write all B-scan files into one zip file.
            compress: For folders, the compression of each file: ``"none"``
                (default), ``"gzip"`` or ``"lzma"``. For zip files, the
                compression inside the zip: ``"deflate"`` (default, opens in
                any zip tool), ``"lzma"`` (smallest) or ``"none"``.
            yaml_arrays: Write rows as YAML lists.
        """
        if is_zip_path(target):
            file_compression = "none"  # the zip file compresses
            zip_compression = compress
        else:
            file_compression = compress or "none"
            zip_compression = None
        suffix = ".yml" + serialization.compression_suffix(file_compression)

        def to_bytes(layer_set: LayerSet) -> bytes:
            return serialization.to_bytes(layer_set, file_compression, yaml_arrays)

        self._save_sets(target, suffix, to_bytes, zip_compression)

    @classmethod
    def load(cls, source: Union[str, Path]) -> LayerVolume:
        """Reads all ``NNN.yml[.gz|.xz]`` files of a folder or zip file,
        ordered by number. In zip files, files in subfolders are found too.

        Raises:
            NotADirectoryError: If ``source`` is neither a folder nor a zip file.
            FileNotFoundError: If it contains no B-scan files.
            ValueError: If a B-scan number occurs twice.
        """
        return cls(cls._read_sets(source, serialization.load_bytes))

    @classmethod
    def load_bscan(cls, source: Union[str, Path], idx: int) -> LayerSet:
        """Reads only B-scan ``idx`` (the number in its file name) of a folder
        or zip file, without reading the other B-scans.

        Raises:
            KeyError: If there is no such B-scan.
        """
        return cls._read_sets(source, serialization.load_bytes, only=idx)[0]

    @classmethod
    def load_from_jsonfolder(cls, folder: Union[str, Path]) -> LayerVolume:
        """Reads a folder of ``NNN.json`` files with ``height`` and
        ``layers: [{name, positions, valid}]``."""
        volume = cls()
        paths = sorted(Path(folder).glob("*.json"), key=lambda path: int(path.stem))
        for path in paths:
            data = json.loads(path.read_text())
            layer_set = LayerSet({"height": data["height"], "idx": int(path.stem)})
            for layer in data["layers"]:
                layer_set.add_layer_raw(
                    layer["name"], layer["positions"], layer["valid"]
                )
            volume.add_layer_set(layer_set)
        return volume
