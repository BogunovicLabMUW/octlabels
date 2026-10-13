import numpy as np
import pytest

from octlabels import AScanRegion, Layer, LayerSet, uncertainty_from_std


def test_width_is_enforced_across_sections(layer_set):
    with pytest.raises(ValueError, match="does not match"):
        layer_set.add_layer_raw("X", np.zeros(5))
    with pytest.raises(ValueError, match="does not match"):
        layer_set.add_region_raw("X", np.zeros(5))


def test_group_is_a_mapping(layer_set):
    assert list(layer_set.layers) == ["ILM", "BM"]
    assert "ILM" in layer_set and "DRUSEN" in layer_set.regions
    assert layer_set[1] is layer_set["BM"] is layer_set.layers["BM"]
    del layer_set.layers["BM"]
    assert len(layer_set) == 1
    with pytest.raises(KeyError, match="Layer 'BM' not found"):
        layer_set.layers["BM"]


def test_group_rejects_wrong_kind_and_mismatched_name(layer_set):
    with pytest.raises(TypeError):
        layer_set.regions["ILM"] = layer_set.layers["ILM"]
    with pytest.raises(ValueError, match="differs from name"):
        layer_set.layers["X"] = Layer("Y", np.zeros(8))


def test_rename_keeps_order_and_rejects_collisions(layer_set):
    layer_set.layers.rename({"ILM": "Inner"})
    assert list(layer_set.layers) == ["Inner", "BM"]
    assert layer_set.layers["Inner"].name == "Inner"
    with pytest.raises(ValueError, match="duplicate"):
        layer_set.layers.rename({"Inner": "BM"})


def test_ensure_creates_missing_with_all_properties(layer_set):
    result = layer_set.ensure(["BM", "RPE"], region_names=["IRF"])
    assert list(result.layers) == ["BM", "RPE"]
    assert list(result.regions) == ["IRF"]
    rpe = result.layers["RPE"]
    assert set(rpe.properties) == {"confidence", "uncertainty", "snr"}
    assert not rpe.confidence.any()
    assert result.uncertainty_to_array().shape == (2, 8)
    assert list(layer_set.ensure(["ILM"]).regions) == ["DRUSEN"]


def test_subset(layer_set):
    result = layer_set.subset(["BM"])
    assert list(result.layers) == ["BM"] and list(result.regions) == ["DRUSEN"]
    with pytest.raises(KeyError):
        layer_set.subset(["RPE"])


def test_copy_is_deep(layer_set):
    copy = layer_set.copy()
    copy.layers["ILM"].values[0] = 99
    copy.info["idx"] = 5
    assert layer_set.layers["ILM"].values[0] == 5
    assert layer_set.info["idx"] == 0


def test_slicing_crops_every_array(layer_set):
    for result in (layer_set[2:5], layer_set[:, 2:5]):
        assert result.width == 3
        assert result.layers["ILM"].properties["snr"].shape == (3,)
        assert result.regions["DRUSEN"].values.tolist() == [1, 1, 1]


def test_arrays(layer_set):
    values, confidence = layer_set.to_array()
    assert values.shape == confidence.shape == (2, 8)
    assert values.dtype == np.float32 and confidence.dtype == np.uint8
    layer_set.values_from_array(values + 1)
    assert layer_set.layers["ILM"].values[0] == 6
    assert layer_set.layers.stack("snr").shape == (2, 8)
    assert layer_set.layers.stack("snr", names=["BM"]).shape == (1, 8)
    with pytest.raises(KeyError, match="no property 'grader'"):
        layer_set.layers.stack("grader")


def test_values_to_array_replaces_nan(layer_set):
    layer_set.layers["ILM"].values[0] = np.nan
    assert layer_set.values_to_array()[0, 0] == 0


def test_from_array():
    values = np.arange(6, dtype=float).reshape(2, 3)
    ls = LayerSet.from_array(
        values, uncertainty=uncertainty_from_std(np.ones((2, 3))), info={"height": 9}
    )
    assert list(ls.layers) == ["Layer 0", "Layer 1"]
    assert ls.width == 3 and ls.height == 9
    assert ls.confidence_to_array().all()
    assert LayerSet.from_string(ls.to_string()).layers["Layer 1"].values.tolist() == [
        3,
        4,
        5,
    ]
    with pytest.raises(TypeError, match="uncertainty_from_std"):
        LayerSet.from_array(values, uncertainty_array=np.ones((2, 3)))


def test_empty():
    ls = LayerSet.empty(["A", "B"], 4, {"height": 9}, region_names=["IRF"])
    assert ls.shape == (2, 9, 4)
    assert list(ls.regions) == ["IRF"]


def test_validate(layer_set):
    layer_set.validate()
    layer_set.layers["BM"].values[0] = 21
    with pytest.raises(ValueError, match="outside"):
        layer_set.validate()
    del layer_set.info["height"]
    with pytest.raises(ValueError, match="height"):
        layer_set.validate()


def test_invalid_key_type(layer_set):
    with pytest.raises(TypeError, match="Invalid key type"):
        layer_set[1.5]


def test_map_annotations_rejects_inconsistent_widths(layer_set):
    def crop_layers_only(annotation):
        if isinstance(annotation, Layer):
            return annotation.map_arrays(lambda _, array: array[:4])
        return annotation.copy()

    with pytest.raises(ValueError, match="different widths"):
        layer_set.map_annotations(crop_layers_only)


def test_from_array_rejects_bad_input():
    with pytest.raises(ValueError, match="must be 2D"):
        LayerSet.from_array(np.zeros(4))
    with pytest.raises(ValueError, match="names for 2 layers"):
        LayerSet.from_array(np.zeros((2, 4)), names=["A"])
    with pytest.raises(TypeError, match="'layer_names' was renamed; use names"):
        LayerSet.from_array(np.zeros((2, 4)), layer_names=["A", "B"])


def test_add_layer_and_region_objects():
    ls = LayerSet()
    ls.add_layer(Layer("ILM", [1, 2, 3]))
    ls.add_region(AScanRegion("IRF", [0, 1, 0]))
    assert ls.width == 3
    with pytest.raises(ValueError, match="already exists"):
        ls.add_layer(Layer("ILM", [1, 2, 3]))


def test_update_from_array(layer_set):
    values, confidence = layer_set.to_array()
    uncertainty = np.full((2, 8), 7)
    layer_set.update_from_array(values * 2, confidence * 0, uncertainty)
    assert layer_set.layers["ILM"].values[1] == pytest.approx(11)
    assert not layer_set.confidence_to_array().any()
    assert (layer_set.uncertainty_to_array() == 7).all()
    with pytest.raises(ValueError, match="Expected shape"):
        layer_set.confidence_from_array(np.zeros((3, 8)))


def test_stack_on_empty_group_has_right_shape_and_dtype():
    ls = LayerSet({"width": 5})
    assert ls.regions.stack().shape == (0, 5)
    assert ls.regions.stack().dtype == np.uint8
    assert ls.layers.stack("confidence").dtype == np.uint8
    assert ls.layers.stack("snr").dtype == np.float32


def test_width_zero_layer_set_roundtrips():
    ls = LayerSet.empty(["A"], 0, {"height": 5})
    assert ls.to_array()[0].shape == (1, 0)
    loaded = LayerSet.from_string(ls.to_string())
    assert loaded.width == 0 and len(loaded.layers["A"]) == 0


def test_info_key_order_survives_copy_and_slicing(layer_set):
    assert list(layer_set.copy().info) == ["width", "height", "idx"]
    assert list(layer_set[2:4].info) == ["width", "height", "idx"]


def test_validate_detects_width_mismatch(layer_set):
    layer_set.info["width"] = 9
    with pytest.raises(ValueError, match="has width 8, expected 9"):
        layer_set.validate()


def test_repr_and_str(layer_set):
    assert repr(layer_set).startswith("LayerSet(info={'width': 8")
    assert repr(layer_set.layers) == "AnnotationGroup(Layer, ['ILM', 'BM'])"
    text = str(layer_set)
    assert "--- Layers (2) ---" in text and "--- Regions (1) ---" in text


def test_slicing_without_annotations_keeps_new_width():
    sliced = LayerSet({"width": 10, "height": 5})[2:6]
    assert sliced.width == 4
    assert "Layers" not in str(sliced)


def test_update_from_array_values_only(layer_set):
    confidence = layer_set.confidence_to_array()
    layer_set.update_from_array(np.zeros((2, 8)))
    assert not layer_set.values_to_array().any()
    np.testing.assert_array_equal(layer_set.confidence_to_array(), confidence)


def test_known_info_keys_are_validated(layer_set):
    layer_set.info.update(
        axial_resolution_um=3.87,
        lateral_resolution_um=11,
        laterality="OD",
        source="corrected",
        subject_id="S001",
        custom_key=[1, 2],  # unknown keys are not checked
    )
    layer_set.validate()


@pytest.mark.parametrize(
    "key, value",
    [
        ("width", True),
        ("idx", -1),
        ("axial_resolution_um", 0),
        ("lateral_resolution_um", float("nan")),
        ("bscan_spacing_um", "120"),
        ("laterality", "right"),
        ("source", "human"),
        ("grader", 7),
        ("format_version", 0),
    ],
)
def test_invalid_known_info_values(layer_set, key, value):
    layer_set.info[key] = value
    with pytest.raises(ValueError, match=f"info\\['{key}'\\]"):
        layer_set.validate()


def test_known_info_keys_are_documented():
    from octlabels import INFO_KEYS

    assert all(key.description for key in INFO_KEYS.values())
    assert {"axial_resolution_um", "lateral_resolution_um", "laterality"} <= set(
        INFO_KEYS
    )
