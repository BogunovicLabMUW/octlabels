import numpy as np
import pytest

from octlabels.layers import transforms


def _all_arrays(layer_set):
    for group in layer_set.sections().values():
        for annotation in group.values():
            yield "values", annotation, annotation.values
            for key, array in annotation.properties.items():
                yield key, annotation, array


def test_flip_reverses_every_array(layer_set):
    flipped = transforms.flip(layer_set)
    original = {(a.name, k): v for k, a, v in _all_arrays(layer_set)}
    for key, annotation, array in _all_arrays(flipped):
        np.testing.assert_array_equal(array, original[annotation.name, key][::-1])


@pytest.mark.parametrize("new_width", [4, 8, 16])
def test_resize_horizontal_keeps_arrays_aligned(layer_set, new_width):
    resized = transforms.resize_horizontal(layer_set, new_width)
    assert resized.width == new_width
    for key, annotation, array in _all_arrays(resized):
        assert array.shape == (new_width,), (annotation.name, key)
        source = layer_set.sections()["layers"].get(annotation.name)
        if source is not None:
            assert (
                array.dtype
                == (source.values if key == "values" else source.properties[key]).dtype
            )


def test_resize_horizontal_same_width_is_identity(layer_set):
    resized = transforms.resize_horizontal(layer_set, 8)
    for (k, _, a), (_, _, b) in zip(
        _all_arrays(layer_set), _all_arrays(resized), strict=True
    ):
        np.testing.assert_array_equal(a, b, err_msg=k)


def test_resize_horizontal_confidence_is_conservative(layer_set):
    resized = transforms.resize_horizontal(layer_set, 16)
    # BM is invalid at A-scan 3; both upsampled neighbours must be invalid.
    assert resized.layers["BM"].confidence[6:8].tolist() == [0, 0]


def test_resize_vertical_scales_layer_heights_only(layer_set):
    resized = transforms.resize_vertical(layer_set, 40)
    assert resized.height == 40
    np.testing.assert_allclose(
        resized.layers["ILM"].values, layer_set.layers["ILM"].values * 2
    )
    np.testing.assert_array_equal(
        resized.regions["DRUSEN"].values, layer_set.regions["DRUSEN"].values
    )


def test_spatial_pad_end(layer_set):
    padded = transforms.spatial_pad_end(layer_set, (32, 10), pad_value=-1)
    assert (padded.height, padded.width) == (32, 10)
    assert padded.layers["ILM"].values[-2:].tolist() == [-1, -1]
    assert padded.layers["ILM"].confidence[-2:].tolist() == [0, 0]
    assert padded.layers["ILM"].properties["snr"].shape == (10,)
    assert padded.regions["DRUSEN"].values[-2:].tolist() == [0, 0]


def test_rotate_zero_is_identity(layer_set):
    rotated = transforms.rotate(layer_set, 0.0)
    for (k, _, a), (_, _, b) in zip(
        _all_arrays(layer_set), _all_arrays(rotated), strict=True
    ):
        np.testing.assert_allclose(a, b, err_msg=k)


def test_translation_moves_all_arrays_together(layer_set):
    moved = transforms.affine(layer_set, translation=(2, 1))
    ilm, original = moved.layers["ILM"], layer_set.layers["ILM"]
    np.testing.assert_allclose(ilm.values[2:], original.values[:-2] + 1)
    assert ilm.confidence[:2].tolist() == [0, 0]
    np.testing.assert_array_equal(ilm.uncertainty[2:], original.uncertainty[:-2])
    np.testing.assert_allclose(
        ilm.properties["snr"][2:], original.properties["snr"][:-2]
    )
    assert moved.regions["DRUSEN"].values.tolist() == [0, 0, 0, 0, 1, 1, 1, 0]


def test_points_leaving_the_image_lose_confidence(layer_set):
    moved = transforms.affine(layer_set, translation=(0, 10))
    assert not moved.layers["BM"].confidence.any()  # BM is pushed below height 20


def test_rotation_keeps_layers_and_properties_aligned(layer_set):
    rotated = transforms.rotate(layer_set, np.deg2rad(10))
    for _, _, array in _all_arrays(rotated):
        assert array.shape == (8,)


def test_order_reversing_transform_is_rejected(layer_set):
    with pytest.raises(ValueError, match="order"):
        transforms.rotate(layer_set, np.pi)


def test_folded_boundary_is_marked_invalid_instead_of_failing():
    from octlabels import LayerSet

    ls = LayerSet({"width": 40, "height": 200})
    ls.add_layer_raw("RPE", np.where(np.arange(40) < 20, 100.0, 130.0))  # 30px step
    rotated = transforms.rotate(ls, np.deg2rad(5))
    confidence = rotated.layers["RPE"].confidence
    # The step folds back from x=18.96 to x=17.34: only column 18 is crossed
    # more than once, so only it has no single height.
    assert confidence[18] == 0
    assert confidence[1:18].all() and confidence[19:37].all()


def test_scale_stretches_around_center(layer_set):
    scaled = transforms.scale(layer_set, 2.0, 1.0)
    # Column c comes from A-scan (c - 3.5) / 2 + 3.5; ILM is 5 + 0.5 * x.
    columns = np.arange(8)
    expected = 5 + 0.5 * ((columns - 3.5) / 2 + 3.5)
    np.testing.assert_allclose(scaled.layers["ILM"].values, expected, atol=1e-5)
    assert scaled.layers["ILM"].properties["snr"].shape == (8,)


def test_shear_tilts_heights(layer_set):
    sheared = transforms.shear(layer_set, 0.0, 0.1)
    offset = 0.1 * (np.arange(8) - 3.5)
    np.testing.assert_allclose(
        sheared.layers["ILM"].values, layer_set.layers["ILM"].values + offset, atol=1e-5
    )


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda ls: transforms.resize_horizontal(ls, 0), "new_width"),
        (lambda ls: transforms.resize_vertical(ls, 0), "new_height"),
        (lambda ls: transforms.resize_vertical(_without_height(ls), 10), "height"),
        (lambda ls: transforms.rotate(_without_height(ls), 0.1), "height"),
    ],
)
def test_invalid_arguments(layer_set, call, message):
    with pytest.raises(ValueError, match=message):
        call(layer_set)


def _without_height(layer_set):
    result = layer_set.copy()
    del result.info["height"]
    return result


def test_fully_invalid_layer_is_kept_by_affine(layer_set):
    layer_set.layers["BM"].confidence = np.zeros(8)
    moved = transforms.affine(layer_set, translation=(2, 0))
    np.testing.assert_array_equal(
        moved.layers["BM"].values, layer_set.layers["BM"].values
    )
    assert not moved.layers["BM"].confidence.any()


def test_custom_interpolation_in_subclass():
    from octlabels import Layer, LayerSet

    class SmoothLayer(Layer):
        INTERPOLATION = {**Layer.INTERPOLATION, "uncertainty": "linear"}

    ls = LayerSet({"width": 2, "height": 10})
    ls.add_layer(SmoothLayer("A", [1, 1], uncertainty=[0, 10]))
    resized = transforms.resize_horizontal(ls, 4)
    layer = resized.layers["A"]
    assert type(layer) is SmoothLayer
    assert layer.uncertainty.dtype == np.uint16
    assert layer.uncertainty.tolist() == [0, 2, 8, 10]  # rounded linear values

    class BrokenLayer(Layer):
        INTERPOLATION = {"values": "cubic"}

    ls = LayerSet({"width": 2, "height": 10})
    ls.add_layer(BrokenLayer("A", [1, 2]))
    with pytest.raises(ValueError, match="Unknown interpolation"):
        transforms.resize_horizontal(ls, 4)


def test_nan_values_never_get_confidence(layer_set):
    layer_set.layers["ILM"].values[2] = np.nan
    layer_set.layers["ILM"].confidence[2] = 0
    rotated = transforms.rotate(layer_set, np.deg2rad(3))
    assert np.isfinite(rotated.layers["ILM"].values).all()
    for result in (
        transforms.flip(layer_set),
        transforms.resize_horizontal(layer_set, 16),
    ):
        ilm = result.layers["ILM"]
        assert not (np.isnan(ilm.values) & (ilm.confidence > 0)).any()


@pytest.mark.parametrize("new_width", [1, 2])
def test_strong_downsampling(layer_set, new_width):
    resized = transforms.resize_horizontal(layer_set, new_width)
    for _, _, array in _all_arrays(resized):
        assert array.shape == (new_width,)


def test_layer_with_single_valid_ascan_stays_flat_when_moved(layer_set):
    confidence = np.zeros(8)
    confidence[4] = 1
    layer_set.layers["BM"].confidence = confidence
    moved = transforms.affine(layer_set, translation=(1, 0))
    bm = moved.layers["BM"]
    assert bm.confidence.tolist() == [0, 0, 0, 0, 0, 1, 0, 0]
    assert bm.values[5] == pytest.approx(layer_set.layers["BM"].values[4])


def test_region_without_confidence_is_moved(layer_set):
    layer_set.regions["DRUSEN"].remove_property("confidence")
    moved = transforms.affine(layer_set, translation=(2, 0))
    assert moved.regions["DRUSEN"].values.tolist() == [0, 0, 0, 0, 1, 1, 1, 0]
    assert moved.regions["DRUSEN"].confidence is None


def test_rotation_around_explicit_center_keeps_that_point(layer_set):
    # ILM passes through (2, 6); rotating around that point keeps it there.
    rotated = transforms.rotate(layer_set, np.deg2rad(4), center=(2, 6))
    assert rotated.layers["ILM"].values[2] == pytest.approx(6, abs=1e-5)


@pytest.fixture
def with_resolution(layer_set):
    layer_set.info.update(
        lateral_resolution_um=10.0, axial_resolution_um=4.0, laterality="OD"
    )
    return layer_set


def test_resize_updates_resolutions(with_resolution):
    wide = transforms.resize_horizontal(with_resolution, 16)
    assert wide.info["lateral_resolution_um"] == 5.0
    assert wide.info["axial_resolution_um"] == 4.0
    tall = transforms.resize_vertical(with_resolution, 40)
    assert tall.info["axial_resolution_um"] == 2.0
    assert tall.info["lateral_resolution_um"] == 10.0
    wide.validate()
    tall.validate()


def test_scale_updates_resolutions(with_resolution):
    scaled = transforms.scale(with_resolution, 2.0, 0.5)
    assert scaled.info["lateral_resolution_um"] == 5.0
    assert scaled.info["axial_resolution_um"] == 8.0


@pytest.mark.parametrize(
    "transform",
    [
        transforms.flip,
        lambda ls: transforms.rotate(ls, 0.05),
        lambda ls: transforms.shear(ls, 0.05, 0.0),
        lambda ls: transforms.affine(ls, translation=(1, 1)),
        lambda ls: transforms.spatial_pad_end(ls, (30, 12)),
        lambda ls: ls[2:6],
    ],
)
def test_other_transforms_keep_resolutions_and_laterality(with_resolution, transform):
    info = transform(with_resolution).info
    assert info["lateral_resolution_um"] == 10.0
    assert info["axial_resolution_um"] == 4.0
    assert info["laterality"] == "OD"


def test_resolution_keys_are_optional(layer_set):
    resized = transforms.resize_horizontal(layer_set, 16)
    assert "lateral_resolution_um" not in resized.info


def test_resizing_an_empty_layer_set_keeps_resolution():
    from octlabels import LayerSet

    ls = LayerSet({"width": 0, "lateral_resolution_um": 10.0})
    resized = transforms.resize_horizontal(ls, 4)
    assert resized.width == 4 and resized.info["lateral_resolution_um"] == 10.0
