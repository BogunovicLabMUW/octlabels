"""Geometric transforms of LayerSets, e.g. for data augmentation.

All functions return a new LayerSet. Layers, regions and every property are
transformed together, so they stay A-scan aligned. Each array is resampled
according to [`AScanAnnotation.interpolation`][].

``info["lateral_resolution_um"]`` and ``info["axial_resolution_um"]`` are
updated when a transform changes the pixel size; other info is copied.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from octlabels.layers.annotation import AScanAnnotation, Layer
from octlabels.layers.layer_set import LayerSet

Point = Tuple[float, float]


def _rescaled_info(
    info: Dict[str, Any], lateral: float = 1.0, axial: float = 1.0
) -> Dict[str, Any]:
    """Returns a copy of ``info`` with the pixel sizes multiplied by
    ``lateral`` and ``axial``, where present. A factor of 1 leaves the value
    untouched, so integers stay integers."""
    result = dict(info)
    if lateral != 1.0 and "lateral_resolution_um" in result:
        result["lateral_resolution_um"] *= lateral
    if axial != 1.0 and "axial_resolution_um" in result:
        result["axial_resolution_um"] *= axial
    return result


def _resample(array: np.ndarray, positions: np.ndarray, mode: str) -> np.ndarray:
    """Samples ``array`` at fractional indices ``positions`` (within range)."""
    last = len(array) - 1
    if mode == "linear":
        result = np.interp(positions, np.arange(len(array)), array.astype(np.float64))
        if not np.issubdtype(array.dtype, np.floating):
            result = np.rint(result)
        return result.astype(array.dtype)
    if mode == "nearest":
        return array[np.clip(np.rint(positions).astype(int), 0, last)]
    if mode == "min":
        lower = np.clip(np.floor(positions).astype(int), 0, last)
        upper = np.clip(np.ceil(positions).astype(int), 0, last)
        return np.minimum(array[lower], array[upper])
    raise ValueError(f"Unknown interpolation mode {mode!r}.")


def flip(layer_set: LayerSet) -> LayerSet:
    """Mirrors all annotations along the A-scan axis.

    ``info`` is unchanged, including ``laterality``: whether a mirrored B-scan
    resembles the other eye depends on the scan direction, which is unknown.
    """

    def mirror(annotation: AScanAnnotation) -> AScanAnnotation:
        return annotation.map_arrays(lambda key, array: array[::-1])

    return layer_set.map_annotations(mirror)


def resize_horizontal(layer_set: LayerSet, new_width: int) -> LayerSet:
    """Resamples all annotations to ``new_width`` A-scans (pixel-center
    aligned, like image resizing)."""
    if new_width <= 0:
        raise ValueError("new_width must be a positive integer.")
    width = layer_set.width or 0
    # Source position of each new A-scan, mapping pixel centers onto each other.
    positions = (np.arange(new_width) + 0.5) * width / new_width - 0.5
    positions = np.clip(positions, 0, max(width - 1, 0))

    def resample(annotation: AScanAnnotation) -> AScanAnnotation:
        def resample_array(key: str, array: np.ndarray) -> np.ndarray:
            return _resample(array, positions, annotation.interpolation(key))

        return annotation.map_arrays(resample_array)

    if width:
        info = _rescaled_info(layer_set.info, lateral=width / new_width)
    else:
        info = dict(layer_set.info)
    info["width"] = new_width
    return layer_set.map_annotations(resample, info=info)


def resize_vertical(layer_set: LayerSet, new_height: int) -> LayerSet:
    """Scales layer heights to ``new_height`` pixels per A-scan."""
    if new_height <= 0:
        raise ValueError("new_height must be a positive integer.")
    if not layer_set.height:
        raise ValueError("LayerSet has no info['height'].")
    factor = new_height / layer_set.height

    def scale_heights(key: str, array: np.ndarray) -> np.ndarray:
        if key == "values":
            return array * factor
        return array

    def scale_layer(annotation: AScanAnnotation) -> AScanAnnotation:
        if isinstance(annotation, Layer):
            return annotation.map_arrays(scale_heights)
        return annotation.copy()  # regions have no heights

    info = _rescaled_info(layer_set.info, axial=1 / factor)
    info["height"] = new_height
    return layer_set.map_annotations(scale_layer, info=info)


def spatial_pad_end(
    layer_set: LayerSet, new_size: Tuple[int, int], pad_value: float = 0
) -> LayerSet:
    """Pads at the end to at least ``new_size = (height, width)``.

    Layer values are padded with ``pad_value``; everything else with 0, so
    padded A-scans have zero confidence.
    """
    height, width = new_size
    current = layer_set.width or 0
    pad = max(width - current, 0)

    def pad_arrays(annotation: AScanAnnotation) -> AScanAnnotation:
        def pad_array(key: str, array: np.ndarray) -> np.ndarray:
            if key == "values" and isinstance(annotation, Layer):
                fill = pad_value
            else:
                fill = 0
            return np.pad(array, (0, pad), constant_values=fill)

        return annotation.map_arrays(pad_array)

    info = {
        **layer_set.info,
        "width": max(current, width),
        "height": max(layer_set.height or 0, height),
    }
    return layer_set.map_annotations(pad_arrays, info=info)


def affine(
    layer_set: LayerSet,
    angle_radians: float = 0.0,
    shear: Point = (0.0, 0.0),
    translation: Point = (0.0, 0.0),
    scale: Point = (1.0, 1.0),
    center: Optional[Point] = None,
) -> LayerSet:
    """Applies ``p' = S (H R (p - c) + t) + c`` to all layer points.

    ``R`` rotates by ``angle_radians`` (counter-clockwise in x/y coordinates),
    ``H`` shears, ``t`` translates and ``S`` scales, around ``center``
    (default: the image center). The transform must keep A-scans in order
    (e.g. ``|angle| < 90°``).

    Layers: invalid A-scans (confidence 0) are bridged by linear
    interpolation before transforming; afterwards confidence is 0 where the
    result lies outside the image. Regions and other annotations without
    heights move with the transformed center row.

    The resolutions in ``info`` are divided by ``scale``; rotation and shear
    leave them unchanged (exact for small augmentation angles only).
    """
    width, height = layer_set.width, layer_set.height
    if not width or not height:
        raise ValueError("LayerSet needs info['width'] and info['height'].")
    if center is None:
        center = ((width - 1) / 2, (height - 1) / 2)
    cos, sin = math.cos(angle_radians), math.sin(angle_radians)
    rotation = np.array([[cos, -sin], [sin, cos]])
    shearing = np.array([[1.0, shear[0]], [shear[1], 1.0 + shear[0] * shear[1]]])
    scaling = np.diag(scale)
    # Column vectors, to apply to points of shape [2, num_points].
    c = np.asarray(center, dtype=np.float64).reshape(2, 1)
    t = np.asarray(translation, dtype=np.float64).reshape(2, 1)

    def transform_points(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        points = np.stack([x, y])
        points = scaling @ (shearing @ rotation @ (points - c) + t) + c
        return points[0], points[1]

    def transform(annotation: AScanAnnotation) -> AScanAnnotation:
        return _transform_annotation(
            annotation, transform_points, width, height, center_y=center[1]
        )

    info = _rescaled_info(layer_set.info, lateral=1 / scale[0], axial=1 / scale[1])
    return layer_set.map_annotations(transform, info=info)


def _transform_annotation(
    annotation: AScanAnnotation,
    transform_points: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    width: int,
    height: int,
    center_y: float,
) -> AScanAnnotation:
    x = np.arange(len(annotation), dtype=np.float64)
    if not isinstance(annotation, Layer):
        new_x, _ = transform_points(x, np.full_like(x, center_y))
        return _warp(annotation, new_x, width)
    valid = annotation.confidence > 0
    if not valid.any():
        return annotation.copy()
    new_x, new_y = transform_points(x, _bridge_invalid(annotation.values, valid))
    result = _warp(annotation, new_x, width, new_y)
    result.confidence[(result.values <= 0) | (result.values >= height)] = 0
    return result


def _bridge_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replaces invalid entries: linear interpolation between valid ones, and
    linear extrapolation of the first and last two valid ones at the ends."""
    x = np.arange(len(values), dtype=np.float64)
    y = values.astype(np.float64)
    known = np.flatnonzero(valid)
    result = np.interp(x, known, y[known])  # constant beyond the ends
    if len(known) < 2:
        return result

    first, second = known[0], known[1]
    before = x < first
    slope = (y[second] - y[first]) / (second - first)
    result[before] = y[first] + slope * (x[before] - first)

    last, second_last = known[-1], known[-2]
    after = x > last
    slope = (y[last] - y[second_last]) / (last - second_last)
    result[after] = y[last] + slope * (x[after] - last)
    return result


def _warp(
    annotation: AScanAnnotation,
    new_x: np.ndarray,
    width: int,
    new_values: Optional[np.ndarray] = None,
) -> AScanAnnotation:
    """Moves A-scan ``i`` to column ``new_x[i]`` and resamples onto the integer
    columns covered by the result; uncovered columns become 0.

    A steep boundary can fold back after rotation or shear, so that a column
    is crossed more than once. Such columns have no single height; they are
    filled from the forward-moving part and get confidence 0.
    """
    if new_x[-1] <= new_x[0]:
        raise ValueError("Transform reverses A-scan order (e.g. |angle| >= 90°).")
    # Keep the points that move forward past every earlier point; the points
    # of a fold are skipped.
    furthest_so_far = np.maximum.accumulate(new_x)
    keep = np.ones(len(new_x), dtype=bool)
    keep[1:] = new_x[1:] > furthest_so_far[:-1]
    kept_x = new_x[keep]
    first = max(math.ceil(kept_x[0]), 0)
    last = min(math.floor(kept_x[-1]), width - 1)
    columns = np.arange(first, last + 1)
    sources = np.interp(columns, kept_x, np.flatnonzero(keep).astype(np.float64))

    def place(key: str, array: np.ndarray) -> np.ndarray:
        result = np.zeros(width, dtype=array.dtype)
        if key == "values" and new_values is not None:
            result[columns] = np.interp(columns, kept_x, new_values[keep])
        else:
            result[columns] = _resample(array, sources, annotation.interpolation(key))
        return result

    result = annotation.map_arrays(place)
    if result.confidence is not None:
        result.confidence[_crossings(new_x, width) > 1] = 0
    return result


def _crossings(new_x: np.ndarray, width: int) -> np.ndarray:
    """Counts how often the polyline through ``new_x`` crosses each column."""
    # Segment i spans the columns ceil(low[i]) .. ceil(high[i]) - 1. Adding +1
    # at each start and -1 at each stop, a running sum gives the count.
    low = np.minimum(new_x[:-1], new_x[1:])
    high = np.maximum(new_x[:-1], new_x[1:])
    start = np.clip(np.ceil(low).astype(int), 0, width)
    stop = np.clip(np.ceil(high).astype(int), 0, width)
    changes = np.zeros(width + 1, dtype=np.int64)
    np.add.at(changes, start, 1)
    np.add.at(changes, stop, -1)
    return np.cumsum(changes)[:width]


def rotate(
    layer_set: LayerSet, angle_radians: float, center: Optional[Point] = None
) -> LayerSet:
    """Rotates around ``center`` (default: image center), see [`affine`][]."""
    return affine(layer_set, angle_radians=angle_radians, center=center)


def shear(
    layer_set: LayerSet, x_shear: float, y_shear: float, center: Optional[Point] = None
) -> LayerSet:
    """Shears around ``center`` (default: image center), see [`affine`][]."""
    return affine(layer_set, shear=(x_shear, y_shear), center=center)


def scale(
    layer_set: LayerSet, x_scale: float, y_scale: float, center: Optional[Point] = None
) -> LayerSet:
    """Scales around ``center`` (default: image center), see [`affine`][]."""
    return affine(layer_set, scale=(x_scale, y_scale), center=center)
