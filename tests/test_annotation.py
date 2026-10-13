import numpy as np
import pytest

from octlabels import Layer, uncertainty_from_std


def test_layer_defaults_confidence_to_ones():
    layer = Layer("ILM", [1.5, 2.0])
    assert layer.confidence.tolist() == [1, 1]
    assert layer.uncertainty is None


def test_properties_must_be_aligned():
    with pytest.raises(ValueError, match="expected \\(3,\\)"):
        Layer("ILM", [1, 2, 3], confidence=[1, 1])
    layer = Layer("ILM", [1, 2, 3])
    with pytest.raises(ValueError):
        layer.set_property("snr", [1.0, 2.0])
    with pytest.raises(ValueError, match="Create a new annotation"):
        layer.values = [1, 2]


def test_custom_property_keeps_dtype_and_name_is_checked():
    layer = Layer("ILM", [1, 2], snr=np.array([0.5, 1.5], dtype=np.float64))
    assert layer.properties["snr"].dtype == np.float64
    for bad in ("values", "1x", "with space"):
        with pytest.raises(ValueError, match="Invalid property name"):
            layer.set_property(bad, [0, 0])


def test_fixed_property_dtypes():
    layer = Layer("ILM", [1, 2], confidence=[1, 0], uncertainty=[5, 999])
    assert layer.confidence.dtype == np.uint8
    assert layer.uncertainty.dtype == np.uint16
    assert layer.values.dtype == np.float32


def test_copy_is_deep_and_keeps_type():
    layer = Layer("ILM", [1, 2], snr=[0.1, 0.2])
    copy = layer.copy()
    copy.values[0] = 99
    copy.properties["snr"][0] = 99
    assert type(copy) is Layer
    assert layer.values[0] == 1 and layer.properties["snr"][0] == pytest.approx(0.1)


def test_input_arrays_are_copied():
    values = np.array([1.0, 2.0], dtype=np.float32)
    layer = Layer("ILM", values)
    values[0] = 99
    assert layer.values[0] == 1


def test_zeros_includes_required_and_requested_properties():
    layer = Layer.zeros("X", 3, {"uncertainty": np.uint16})
    assert layer.confidence.tolist() == [0, 0, 0]
    assert layer.uncertainty.tolist() == [0, 0, 0]


def test_required_property_cannot_be_removed():
    with pytest.raises(ValueError):
        Layer("ILM", [1]).remove_property("confidence")


def test_uncertainty_from_std_matches_legacy_encoding():
    assert uncertainty_from_std([0.5, 1, 5, 50, 100, 200]).tolist() == [
        0,
        0,
        40,
        494,
        999,
        999,
    ]


def test_values_must_be_1d_and_numeric():
    with pytest.raises(ValueError, match="must be 1D"):
        Layer("ILM", [[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="must be numeric"):
        Layer("ILM", ["a", "b"])


def test_confidence_and_uncertainty_setters_convert_dtype():
    layer = Layer("ILM", [1, 2])
    layer.confidence = [0, 1]
    layer.uncertainty = [5, 999]
    assert layer.confidence.dtype == np.uint8
    assert layer.uncertainty.dtype == np.uint16
    with pytest.raises(ValueError):
        layer.uncertainty = [1, 2, 3]


def test_remove_custom_property():
    layer = Layer("ILM", [1, 2], snr=[0.1, 0.2])
    layer.remove_property("snr")
    assert set(layer.properties) == {"confidence"}
    with pytest.raises(KeyError):
        layer.remove_property("snr")


def test_properties_view_is_read_only():
    layer = Layer("ILM", [1, 2])
    with pytest.raises(TypeError):
        layer.properties["snr"] = np.zeros(2)


def test_repr_and_str():
    layer = Layer("ILM", np.arange(12), snr=np.zeros(12))
    assert repr(layer) == (
        "Layer(name='ILM', values=[12, float32], confidence=[12, uint8], "
        "snr=[12, float64])"
    )
    text = str(layer)
    assert text.startswith("Layer: ILM (12 A-scans)")
    assert "snr" in text and "..." in text


def test_annotation_without_values_is_empty():
    layer = Layer("ILM")
    assert len(layer) == 0 and layer.confidence.shape == (0,)
