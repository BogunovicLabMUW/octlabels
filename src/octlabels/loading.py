"""Loading and saving with one call: files, volumes and plain arrays.

Layer files are ``NNN.yml[.gz|.xz]``, lesion files ``NNN.png``; a volume is a
folder or ``.zip`` file of such files.
"""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Type, Union

import numpy as np
from numpy.typing import ArrayLike

from octlabels.layers.layer_set import LayerSet
from octlabels.layers.layer_volume import LayerVolume
from octlabels.lesions.lesion_set import LesionSet
from octlabels.lesions.lesion_volume import LesionVolume
from octlabels.volume import is_zip_path

PathLike = Union[str, Path]
VolumeClass = Union[Type[LayerVolume], Type[LesionVolume]]


class LayerArrays(NamedTuple):
    """Layer names with their stacked values and confidences.

    Attributes:
        names: Layer names, in array order.
        values: Boundary heights (float32, NaN replaced by 0).
            Shape: [num_layers, width] or [num_bscans, num_layers, width].
        confidence: Confidences (uint8), same shape as ``values``.
    """

    names: List[str]
    values: np.ndarray
    confidence: np.ndarray


class LesionArrays(NamedTuple):
    """Lesion names with their stacked masks.

    Attributes:
        names: Lesion names, in array order.
        masks: Masks (uint8 0/1). Shape: [num_lesions, height, width] or
            [num_bscans, num_lesions, height, width].
    """

    names: List[str]
    masks: np.ndarray


def _volume_class(path: Path) -> VolumeClass:
    """Returns LayerVolume or LesionVolume, depending on the files that a
    folder or zip file contains."""
    if path.is_dir():
        file_names = [child.name for child in path.iterdir()]
    elif path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            file_names = [PurePosixPath(name).name for name in archive.namelist()]
    else:
        raise NotADirectoryError(f"'{path}' is neither a folder nor a zip file.")

    has_layers = _contains_files_of(LayerVolume, file_names)
    has_lesions = _contains_files_of(LesionVolume, file_names)
    if has_layers and has_lesions:
        raise ValueError(
            f"'{path}' contains layer and lesion files; load them with "
            "LayerVolume.load or LesionVolume.load."
        )
    if has_layers:
        return LayerVolume
    if has_lesions:
        return LesionVolume
    raise FileNotFoundError(f"No layer or lesion files in '{path}'.")


def _contains_files_of(volume_class: VolumeClass, file_names: List[str]) -> bool:
    return any(volume_class.FILE_PATTERN.match(name) for name in file_names)


def load(path: PathLike) -> Union[LayerSet, LesionSet, LayerVolume, LesionVolume]:
    """Loads annotations, choosing the type from the path.

    Returns:
        A [`LesionSet`][] for a ``.png`` file, a [`LayerSet`][] for a
        ``.yml[.gz|.xz]`` file, and a [`LayerVolume`][] or
        [`LesionVolume`][] for a folder or ``.zip`` file of such files.
    """
    path = Path(path)
    if path.is_dir() or is_zip_path(path):
        return _volume_class(path).load(path)
    if path.suffix.lower() == ".png":
        return LesionSet.load(path)
    return LayerSet.load(path)


def load_bscan(volume: PathLike, idx: int) -> Union[LayerSet, LesionSet]:
    """Loads B-scan ``idx`` (the number in its file name) from a volume folder
    or ``.zip`` file, without reading the other B-scans."""
    path = Path(volume)
    return _volume_class(path).load_bscan(path, idx)


def load_layer_arrays(
    path: PathLike, names: Optional[Sequence[str]] = None
) -> LayerArrays:
    """Loads layers as arrays.

    Args:
        path: A B-scan file, or a volume folder or ``.zip`` file.
        names: Layers to return, in this order; missing ones are zero
            (value 0, confidence 0). Defaults to the layers in the file(s),
            which for a volume must be the same in every B-scan.

    Returns:
        Names, values and confidence; arrays of shape [num_layers, width] for
        a B-scan and [num_bscans, num_layers, width] for a volume.
    """
    loaded = load(path)
    if isinstance(loaded, LayerVolume):
        names = loaded._common_names(names, lambda layer_set: layer_set.layers)
    elif isinstance(loaded, LayerSet):
        if names is None:
            names = list(loaded.layers)
    else:
        raise TypeError(f"'{path}' contains lesions; use load_lesion_arrays.")
    values, confidence = loaded.to_array(names)
    return LayerArrays(list(names), values, confidence)


def load_lesion_arrays(
    path: PathLike, names: Optional[Sequence[str]] = None
) -> LesionArrays:
    """Loads lesion masks as arrays.

    Args:
        path: A B-scan ``.png`` file, or a volume folder or ``.zip`` file.
        names: Lesions to return, in this order; missing ones are empty
            masks. Defaults to the lesions in the file(s), which for a volume
            must be the same in every B-scan.

    Returns:
        Names and masks; masks of shape [num_lesions, height, width] for a
        B-scan and [num_bscans, num_lesions, height, width] for a volume.
    """
    loaded = load(path)
    if isinstance(loaded, LesionVolume):
        names = loaded._common_names(names, lambda lesion_set: lesion_set.lesions)
    elif isinstance(loaded, LesionSet):
        if names is None:
            names = list(loaded.lesions)
    else:
        raise TypeError(f"'{path}' contains layers; use load_layer_arrays.")
    return LesionArrays(list(names), loaded.to_array(names))


def save_layer_arrays(
    path: PathLike,
    values: ArrayLike,
    confidence: Optional[ArrayLike] = None,
    names: Optional[Sequence[str]] = None,
    info: Optional[Mapping[str, Any]] = None,
    **properties: ArrayLike,
) -> None:
    """Saves layer arrays as a B-scan file or a volume.

    Args:
        path: For 2D ``values``, a ``.yml[.gz|.xz]`` file; for 3D ``values``,
            a volume folder or ``.zip`` file.
        values: Boundary heights.
            Shape: [num_layers, width] or [num_bscans, num_layers, width].
        confidence: Same shape; defaults to all ones.
        names: Layer names; defaults to ``"Layer 0"``, ...
        info: Metadata, e.g. ``{"height": 496}``.
        **properties: Further properties of the same shape, e.g. ``uncertainty``.
    """
    values = np.asarray(values)
    if values.ndim == 2:
        LayerSet.from_array(values, confidence, names, info, **properties).save(path)
    elif values.ndim == 3:
        LayerVolume.from_array(values, confidence, names, info, **properties).save(path)
    else:
        raise ValueError(f"values must be 2D or 3D, got shape {values.shape}.")


def save_lesion_arrays(
    path: PathLike,
    masks: ArrayLike,
    names: Optional[Sequence[str]] = None,
    main_lesion_names: Sequence[str] = (),
    info: Optional[Mapping[str, Any]] = None,
) -> None:
    """Saves lesion masks as a B-scan PNG or a volume.

    Args:
        path: For 3D ``masks``, a ``.png`` file; for 4D ``masks``, a volume
            folder or ``.zip`` file.
        masks: Masks, nonzero = present. Shape: [num_lesions, height, width]
            or [num_bscans, num_lesions, height, width].
        names: Lesion names; defaults to ``"Lesion 0"``, ...
        main_lesion_names: Which lesions are main (mutually exclusive) lesions;
            all others are overlays.
        info: Metadata for the B-scans.
    """
    masks = np.asarray(masks)
    if masks.ndim == 3:
        LesionSet.from_array(masks, names, info, main_lesion_names).save(path)
    elif masks.ndim == 4:
        LesionVolume.from_array(masks, names, info, main_lesion_names).save(path)
    else:
        raise ValueError(f"masks must be 3D or 4D, got shape {masks.shape}.")
