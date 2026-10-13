"""Biomarker regions (AScanRegion): presence per A-scan, next to the layers."""

import numpy as np
import pytest

from octlabels import AScanRegion, LayerSet, LayerVolume
from octlabels.layers import transforms
from octlabels.layers.serialization import dumps, loads


@pytest.fixture
def regions_only():
    """A B-scan with two regions and no layers."""
    ls = LayerSet({"width": 8, "height": 20, "idx": 0})
    ls.add_region_raw("DRUSEN", [0, 0, 1, 1, 1, 0, 0, 0], confidence=[200] * 8)
    ls.add_region_raw("IRF", [0, 0, 0, 0, 0, 1, 1, 0], uncertainty=np.arange(8) * 100)
    return ls


# --- The AScanRegion class ----------------------------------------------------


@pytest.mark.parametrize(
    "values", [[0, 1, 1], np.array([True, False]), np.array([0.0, 1.0]), []]
)
def test_accepts_presence_flags_of_any_numeric_type(values):
    region = AScanRegion("IRF", values)
    assert region.values.dtype == np.uint8
    assert region.values.tolist() == [int(v) for v in values]


@pytest.mark.parametrize("values", [[0, 2], [0, 0.5], [-1, 1], [1, np.nan]])
def test_rejects_anything_but_0_and_1(values):
    with pytest.raises(ValueError, match="Region 'IRF': values must be 0 or 1"):
        AScanRegion("IRF", values)


def test_values_must_be_1d():
    with pytest.raises(ValueError, match="must be 1D"):
        AScanRegion("IRF", [[0, 1], [1, 0]])


def test_confidence_is_optional_unlike_for_layers():
    region = AScanRegion("IRF", [0, 1, 1])
    assert region.confidence is None and region.properties == {}
    region.confidence = [1, 0, 1]
    assert region.confidence.dtype == np.uint8
    region.remove_property("confidence")  # allowed: not required for regions
    assert region.confidence is None


def test_properties_are_aligned_and_typed():
    region = AScanRegion(
        "DRUSEN", [0, 1], confidence=[9, 9], uncertainty=[0, 999], grader=[1, 2]
    )
    assert region.uncertainty.dtype == np.uint16
    assert region.properties["grader"].tolist() == [1, 2]
    with pytest.raises(ValueError, match="expected \\(2,\\)"):
        region.set_property("snr", [0.1, 0.2, 0.3])


def test_interpolation_modes():
    region = AScanRegion(
        "IRF", [0, 1], confidence=[1, 1], snr=[0.5, 1.5], grader=[1, 2]
    )
    assert region.interpolation("values") == "nearest"  # stays 0 or 1
    assert region.interpolation("confidence") == "min"
    assert region.interpolation("snr") == "linear"
    assert region.interpolation("grader") == "nearest"


def test_copy_map_arrays_and_zeros_keep_the_class():
    region = AScanRegion("IRF", [0, 1, 1], confidence=[5, 5, 5])
    copy = region.copy()
    copy.values[0] = 1
    assert type(copy) is AScanRegion and region.values[0] == 0
    reversed_region = region.map_arrays(lambda key, array: array[::-1])
    assert type(reversed_region) is AScanRegion
    assert reversed_region.values.tolist() == [1, 1, 0]
    empty = AScanRegion.zeros("PED", 4, {"confidence": np.uint8})
    assert type(empty) is AScanRegion
    assert empty.values.tolist() == [0] * 4 and empty.confidence.tolist() == [0] * 4


def test_zeros_without_properties_has_no_confidence():
    assert AScanRegion.zeros("PED", 3).confidence is None


def test_repr_and_str_name_the_class():
    region = AScanRegion("IRF", [0, 1])
    assert repr(region) == "AScanRegion(name='IRF', values=[2, uint8])"
    assert str(region).startswith("AScanRegion: IRF (2 A-scans)")


# --- Regions in a LayerSet ----------------------------------------------------


def test_invalid_values_are_rejected_on_assignment():
    region = AScanRegion("IRF", [0, 1, 0])
    with pytest.raises(ValueError, match="0 or 1"):
        region.values = [0, 2, 0]
    region.values = [1, 1, 0]
    assert region.values.tolist() == [1, 1, 0]


def test_regions_and_layers_are_separate(layer_set):
    layer_set.add_region_raw("ILM", np.zeros(8))  # same name as a layer
    assert layer_set["ILM"] is layer_set.layers["ILM"]
    assert isinstance(layer_set.regions["ILM"], AScanRegion)
    with pytest.raises(ValueError, match="already exists"):
        layer_set.add_region_raw("DRUSEN", np.zeros(8))


def test_rename_stack_and_unstack(regions_only):
    regions_only.regions.rename({"IRF": "SRF"})
    assert list(regions_only.regions) == ["DRUSEN", "SRF"]
    stacked = regions_only.regions.stack()
    assert stacked.shape == (2, 8) and stacked.dtype == np.uint8
    regions_only.regions.unstack(1 - stacked)
    assert regions_only.regions["DRUSEN"].values.tolist() == [1, 1, 0, 0, 0, 1, 1, 1]
    with pytest.raises(ValueError, match="0 or 1"):
        regions_only.regions.unstack(stacked * 2)


def test_ensure_creates_missing_regions_as_absent(regions_only):
    result = regions_only.ensure([], region_names=["IRF", "PED"])
    assert list(result.regions) == ["IRF", "PED"]
    ped = result.regions["PED"]
    assert not ped.values.any()
    # Missing regions get every property used by the existing ones, zero-filled.
    assert set(ped.properties) == {"confidence", "uncertainty"}
    assert not ped.confidence.any() and not ped.uncertainty.any()


@pytest.mark.parametrize("yaml_arrays", [False, True])
def test_file_with_only_regions_roundtrips(regions_only, tmp_path, yaml_arrays):
    loaded = loads(dumps(regions_only, yaml_arrays=yaml_arrays))
    assert not loaded.layers
    assert list(loaded.regions) == ["DRUSEN", "IRF"]
    irf = loaded.regions["IRF"]
    np.testing.assert_array_equal(irf.values, regions_only.regions["IRF"].values)
    np.testing.assert_array_equal(irf.uncertainty, np.arange(8) * 100)
    assert irf.confidence is None


def test_volume_with_only_regions(regions_only, tmp_path):
    second = regions_only.copy()
    second.info["idx"] = 1
    LayerVolume([regions_only, second]).save(tmp_path / "vol.zip")
    loaded = LayerVolume.load(tmp_path / "vol.zip")
    assert len(loaded) == 2 and list(loaded[1].regions) == ["DRUSEN", "IRF"]
    single = LayerVolume.load_bscan(tmp_path / "vol.zip", 1)
    assert single.regions["DRUSEN"].values.tolist() == [0, 0, 1, 1, 1, 0, 0, 0]


@pytest.mark.parametrize("new_width", [3, 5, 8, 16, 21])
def test_resizing_keeps_regions_binary_and_in_place(regions_only, new_width):
    resized = transforms.resize_horizontal(regions_only, new_width)
    drusen = resized.regions["DRUSEN"]
    assert set(np.unique(drusen.values)) <= {0, 1}
    assert drusen.values.dtype == np.uint8
    assert drusen.confidence.tolist() == [200] * new_width
    # The region covers the same fraction of the B-scan (A-scans 2-4 of 8).
    present = np.flatnonzero(drusen.values)
    assert present.min() / new_width == pytest.approx(2 / 8, abs=1 / new_width)
    assert (present.max() + 1) / new_width == pytest.approx(5 / 8, abs=1 / new_width)


def test_doubling_the_width_repeats_each_ascan(regions_only):
    resized = transforms.resize_horizontal(regions_only, 16)
    expected = np.repeat(regions_only.regions["IRF"].values, 2)
    np.testing.assert_array_equal(resized.regions["IRF"].values, expected)


def test_rotation_keeps_regions_binary(regions_only):
    rotated = transforms.rotate(regions_only, np.deg2rad(8))
    for region in rotated.regions.values():
        assert set(np.unique(region.values)) <= {0, 1}
        assert len(region) == 8


def test_flip_mirrors_regions_and_their_properties(regions_only):
    flipped = transforms.flip(regions_only)
    irf, original = flipped.regions["IRF"], regions_only.regions["IRF"]
    np.testing.assert_array_equal(irf.values, original.values[::-1])
    np.testing.assert_array_equal(irf.uncertainty, original.uncertainty[::-1])
