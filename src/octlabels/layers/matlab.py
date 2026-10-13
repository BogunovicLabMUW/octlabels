"""Import of layer boundaries from MATLAB ``.mat`` files, as used by the public
Duke OCT datasets.

This module needs optional packages, imported only when a function is called:
``scipy`` for MATLAB v5 files and ``h5py`` for v7.3 (HDF5) files. Install both
with ``pip install "octlabels[matlab]"``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from octlabels.layers.layer_set import LayerSet
from octlabels.layers.layer_volume import LayerVolume

_DUKE_DME_NAMES = (
    "ILM",
    "NFL-GCL",
    "IPL-INL",
    "INL-OPL",
    "OPL-ONL",
    "ISM-ISE",
    "OS-RPE",
    "BM",
)

DUKE_VARIABLES: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    # Duke DME dataset (Chiu et al., 2015), two graders and two algorithms.
    "manualLayers1": ("LAB", _DUKE_DME_NAMES),
    "manualLayers2": ("LAB", _DUKE_DME_NAMES),
    "automaticLayersDME": ("LAB", _DUKE_DME_NAMES),
    "automaticLayersNormal": ("LAB", _DUKE_DME_NAMES),
    # Duke AMD dataset (Farsiu et al., 2014).
    "layerMaps": ("BAL", ("ILM", "RPEDC", "BM")),
}
"""Known boundary variables: dimension order (L = boundaries, A = A-scans, B = B-scans)
and default boundary names.
"""


def _import_optional(module: str, package: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(
            f"Reading .mat files needs the optional package '{package}'. "
            f'Install it with: pip install "octlabels[matlab]" '
            f"(or: pip install {package})"
        ) from error


def read_mat(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Reads all arrays of a ``.mat`` file, in MATLAB dimension order.

    Uses scipy for MATLAB v5 files and h5py for v7.3 (HDF5) files.
    """
    path = Path(path)
    with open(path, "rb") as file:
        header = file.read(128)
    if header.startswith(b"MATLAB 7.3"):
        h5py = _import_optional("h5py", "h5py")
        with h5py.File(path, "r") as file:
            # HDF5 stores MATLAB arrays with reversed dimensions.
            return {
                name: np.asarray(item).T
                for name, item in file.items()
                if isinstance(item, h5py.Dataset)
            }
    scipy_io = _import_optional("scipy.io", "scipy")
    arrays = scipy_io.loadmat(path)
    # Names like "__header__" are file metadata, not variables.
    return {name: value for name, value in arrays.items() if not name.startswith("__")}


def load_duke_mat(
    path: Union[str, Path],
    variable: Optional[str] = None,
    axes: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    height: Optional[int] = None,
    one_based: bool = True,
    skip_unannotated: bool = True,
) -> LayerVolume:
    """Loads boundaries stored as one numeric array in a ``.mat`` file.

    Missing boundaries are NaN in the file; they get value 0 and confidence 0.

    Args:
        path: The ``.mat`` file.
        variable: Name of the boundary array. Defaults to the first known
            variable of [`DUKE_VARIABLES`][] found in the file.
        axes: Dimension order of the array, a permutation of ``"LAB"``
            (L = boundaries, A = A-scans, B = B-scans). Defaults to the order
            of a known variable.
        names: Layer names. Defaults to the known names, or ``"Layer 0"``, ...
        height: Pixels per A-scan. Defaults to the first dimension of an
            ``images`` array in the file, if present.
        one_based: The file stores MATLAB (1-based) row indices; convert them
            to 0-based.
        skip_unannotated: Leave out B-scans without any annotated boundary.
            ``info["idx"]`` always holds the B-scan's index in the file.

    Raises:
        ImportError: If scipy (v5 files) or h5py (v7.3 files) is missing.
        KeyError: If the variable is not in the file; the message lists the
            available variables.
        ValueError: If ``axes`` is missing or invalid.
    """
    arrays = read_mat(path)
    if variable is None:
        variable = _find_known_variable(arrays, path)
    if variable not in arrays:
        raise KeyError(f"'{variable}' not in {path}. Available: {sorted(arrays)}")
    known_axes, known_names = DUKE_VARIABLES.get(variable, (None, None))
    if axes is None:
        axes = known_axes
    if axes is None or sorted(axes) != ["A", "B", "L"]:
        raise ValueError(f"axes must be a permutation of 'LAB', got {axes!r}.")

    # Reorder the dimensions to [B-scans, boundaries (L), A-scans].
    order = [axes.index("B"), axes.index("L"), axes.index("A")]
    boundaries = np.transpose(np.asarray(arrays[variable], dtype=np.float64), order)
    num_bscans, num_layers, width = boundaries.shape
    if names is None:
        if known_names is not None and len(known_names) == num_layers:
            names = known_names
        else:
            names = [f"Layer {i}" for i in range(num_layers)]
    if height is None and "images" in arrays:
        height = int(arrays["images"].shape[0])

    valid = np.isfinite(boundaries)
    if one_based:
        boundaries = boundaries - 1
    values = np.where(valid, boundaries, 0)

    volume = LayerVolume()
    for idx in range(num_bscans):
        if skip_unannotated and not valid[idx].any():
            continue
        info = {"width": width}
        if height:
            info["height"] = height
        info["idx"] = idx
        volume.add_layer_set(LayerSet.from_array(values[idx], valid[idx], names, info))
    return volume


def _find_known_variable(arrays: Dict[str, np.ndarray], path: Union[str, Path]) -> str:
    for name in DUKE_VARIABLES:
        if name in arrays:
            return name
    raise KeyError(
        f"No known boundary variable in {path}; pass variable=. "
        f"Available: {sorted(arrays)}"
    )
