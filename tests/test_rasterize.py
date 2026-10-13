import numpy as np
import pytest

from octlabels import LayerSet
from octlabels.layers import rasterize


def test_segmentation_bands_partition_valid_columns(layer_set):
    masks, confidence = rasterize.generate_2d_segmentation(layer_set)
    assert masks.shape == (3, 20, 8) and confidence.shape == (3, 1, 8)
    fully_valid = confidence.all(axis=0)[0]
    assert (masks.sum(axis=0)[:, fully_valid] == 1).all()
    assert not masks[1, :, 3].any()  # BM invalid at A-scan 3


def test_boundary_maps_roundtrip(layer_set):
    layer_set.info["height"] = np.int64(20)
    maps = rasterize.generate_boundary_maps(layer_set)
    positions, confidence = rasterize.extract_boundaries_from_maps(maps)
    np.testing.assert_array_equal(positions[0], np.round(layer_set["ILM"].values))
    np.testing.assert_array_equal(confidence, layer_set.confidence_to_array())


def test_layer_set_without_layers():
    ls = LayerSet.empty([], 4, {"height": 5})
    masks, confidence = rasterize.generate_2d_segmentation(ls)
    assert masks.shape == (1, 5, 4) and not masks.any() and not confidence.any()
    assert rasterize.generate_boundary_maps(ls).shape == (0, 5, 4)


def test_missing_height_is_rejected(layer_set):
    del layer_set.info["height"]
    with pytest.raises(ValueError, match="height"):
        rasterize.generate_2d_segmentation(layer_set)


def test_extract_boundaries_edge_cases():
    with pytest.raises(ValueError, match="Expected"):
        rasterize.extract_boundaries_from_maps(np.zeros((4, 5)))
    positions, confidence = rasterize.extract_boundaries_from_maps(np.zeros((2, 0, 3)))
    assert positions.shape == confidence.shape == (2, 3)


def test_nan_and_out_of_range_heights(layer_set):
    layer_set.layers["ILM"].values[0] = np.nan
    layer_set.layers["ILM"].confidence[0] = 0
    layer_set.layers["BM"].values[1] = 99  # below the image
    maps = rasterize.generate_boundary_maps(layer_set)
    assert not maps[0, :, 0].any()
    assert maps[1, 19, 1] == 1  # clipped to the last row
