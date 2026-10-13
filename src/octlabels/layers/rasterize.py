"""Conversion between layer boundaries and 2D maps for segmentation models."""

from __future__ import annotations

import numbers
from typing import Tuple

import numpy as np

from octlabels.layers.layer_set import LayerSet


def _image_size(layer_set: LayerSet) -> Tuple[int, int]:
    return (
        _positive_int(layer_set.height, "height"),
        _positive_int(layer_set.width, "width"),
    )


def _positive_int(value: object, key: str) -> int:
    if not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"info['{key}'] must be a positive integer, got {value!r}.")
    return int(value)


def _rounded_heights(layer_set: LayerSet, height: int) -> np.ndarray:
    """Layer heights rounded and clipped to [0, height]. Shape: [num_layers, width]."""
    values = np.round(layer_set.values_to_array()).astype(np.int64)
    return np.clip(values, 0, height)


def generate_2d_segmentation(layer_set: LayerSet) -> Tuple[np.ndarray, np.ndarray]:
    """Converts layer boundaries into one binary mask per band.

    Band 0 lies above the first layer, band ``i`` between layers ``i-1`` and
    ``i``, and band ``N`` below the last layer. A column of a band is only
    filled where the bounding layers have confidence > 0.

    Returns:
        ``(masks, confidence)``: masks of shape [N + 1, height, width] and
        band confidence of shape [N + 1, 1, width], both uint8 (0 or 1).
    """
    height, width = _image_size(layer_set)
    num_layers = len(layer_set.layers)
    if num_layers == 0:
        return (
            np.zeros((1, height, width), dtype=np.uint8),
            np.zeros((1, 1, width), dtype=np.uint8),
        )
    boundaries = _rounded_heights(layer_set, height)  # [num_layers, width]
    valid = layer_set.confidence_to_array() > 0  # [num_layers, width]

    # Band i covers the rows top[i] <= row < bottom[i]. The first band starts
    # at the top of the image and the last one ends at the bottom.
    image_top = np.zeros((1, width), dtype=np.int64)
    image_bottom = np.full((1, width), height, dtype=np.int64)
    top = np.concatenate([image_top, boundaries])  # [num_layers + 1, width]
    bottom = np.concatenate([boundaries, image_bottom])

    # A band is valid where the boundaries enclosing it are valid; the outer
    # bands have only one such boundary.
    inner_valid = valid[:-1] & valid[1:]
    band_valid = np.concatenate([valid[:1], inner_valid, valid[-1:]])

    rows = np.arange(height)[None, :, None]  # [1, height, 1]
    masks = (rows >= top[:, None, :]) & (rows < bottom[:, None, :])
    confidence = band_valid[:, None, :].astype(np.uint8)
    return masks.astype(np.uint8) * confidence, confidence


def generate_boundary_maps(layer_set: LayerSet) -> np.ndarray:
    """Draws each layer as a one-pixel line where its confidence is > 0.

    Returns:
        uint8 array of shape [num_layers, height, width].
    """
    height, width = _image_size(layer_set)
    maps = np.zeros((len(layer_set.layers), height, width), dtype=np.uint8)
    if not layer_set.layers:
        return maps
    rows = np.clip(_rounded_heights(layer_set, height), 0, height - 1)
    layer_index, column = np.nonzero(layer_set.confidence_to_array() > 0)
    maps[layer_index, rows[layer_index, column], column] = 1
    return maps


def extract_boundaries_from_maps(
    boundary_maps: np.ndarray, threshold: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of [`generate_boundary_maps`][] for (soft) boundary maps.

    Args:
        boundary_maps: Array of shape [num_layers, height, width].
        threshold: A column is confident if its sum exceeds this value.

    Returns:
        ``(positions, confidence)``, both [num_layers, width]: the row of the
        maximum in each column (int64) and 1 where the column is confident
        (uint8).
    """
    boundary_maps = np.asarray(boundary_maps)
    if boundary_maps.ndim != 3:
        raise ValueError(
            f"Expected [num_layers, height, width], got {boundary_maps.shape}."
        )
    num_layers, height, width = boundary_maps.shape
    if height == 0:
        return (
            np.zeros((num_layers, width), dtype=np.int64),
            np.zeros((num_layers, width), dtype=np.uint8),
        )
    positions = np.argmax(boundary_maps, axis=1).astype(np.int64)
    confidence = (boundary_maps.sum(axis=1) > threshold).astype(np.uint8)
    return positions, confidence
