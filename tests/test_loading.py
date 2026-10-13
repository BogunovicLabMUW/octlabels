import zipfile

import numpy as np
import pytest

import octlabels
from octlabels import LayerSet, LayerVolume, LesionSet, LesionVolume


@pytest.fixture
def layer_folder(tmp_path):
    values = np.arange(3)[:, None, None] + np.ones((3, 2, 4))  # B-scan i: i + 1
    octlabels.save_layer_arrays(
        tmp_path / "layers", values, names=["ILM", "BM"], info={"height": 20}
    )
    return tmp_path / "layers"


@pytest.fixture
def lesion_folder(tmp_path):
    masks = np.zeros((3, 2, 5, 4), dtype=np.uint8)
    masks[:, 0, 1] = 1
    masks[2, 1, :, 0] = 1
    octlabels.save_lesion_arrays(
        tmp_path / "lesions", masks, ["SRF", "HRF"], main_lesion_names=["SRF"]
    )
    return tmp_path / "lesions"


def test_load_dispatches_on_path(layer_folder, lesion_folder, tmp_path):
    assert isinstance(octlabels.load(layer_folder), LayerVolume)
    assert isinstance(octlabels.load(layer_folder / "001.yml"), LayerSet)
    assert isinstance(octlabels.load(lesion_folder), LesionVolume)
    assert isinstance(octlabels.load(lesion_folder / "001.png"), LesionSet)
    octlabels.load(lesion_folder).save(tmp_path / "lesions.zip")
    assert isinstance(octlabels.load(tmp_path / "lesions.zip"), LesionVolume)


def test_load_bscan_for_both_kinds(layer_folder, lesion_folder):
    assert octlabels.load_bscan(layer_folder, 1)["ILM"].values[0] == 2
    assert octlabels.load_bscan(lesion_folder, 2)["HRF"].mask[:, 0].all()


def test_layer_arrays_for_bscan_and_volume(layer_folder, tmp_path):
    names, values, confidence = octlabels.load_layer_arrays(layer_folder / "001.yml")
    assert names == ["ILM", "BM"] and values.shape == (2, 4) and (values == 2).all()
    names, values, _ = octlabels.load_layer_arrays(layer_folder / "001.yml", ["BM"])
    assert names == ["BM"] and values.shape == (1, 4)
    assert values.dtype == np.float32 and confidence.dtype == np.uint8
    arrays = octlabels.load_layer_arrays(layer_folder, ["BM", "RPE"])
    assert arrays.names == ["BM", "RPE"] and arrays.values.shape == (3, 2, 4)
    assert not arrays.confidence[:, 1].any()
    octlabels.load(layer_folder).save(tmp_path / "vol.zip", compress="lzma")
    zipped = octlabels.load_layer_arrays(tmp_path / "vol.zip")
    np.testing.assert_array_equal(
        zipped.values, octlabels.load_layer_arrays(layer_folder).values
    )


def test_lesion_arrays_for_bscan_and_volume(lesion_folder):
    names, masks = octlabels.load_lesion_arrays(lesion_folder / "002.png")
    assert names == ["SRF", "HRF"] and masks.shape == (2, 5, 4)
    names, masks = octlabels.load_lesion_arrays(lesion_folder / "002.png", ["HRF"])
    assert names == ["HRF"] and masks.shape == (1, 5, 4)
    arrays = octlabels.load_lesion_arrays(lesion_folder, ["HRF", "PED"])
    assert arrays.names == ["HRF", "PED"] and arrays.masks.shape == (3, 2, 5, 4)
    assert arrays.masks[2, 0, :, 0].all() and not arrays.masks[:, 1].any()


def test_array_functions_check_the_kind(layer_folder, lesion_folder):
    with pytest.raises(TypeError, match="contains layers; use load_layer_arrays"):
        octlabels.load_lesion_arrays(layer_folder)
    with pytest.raises(TypeError, match="contains lesions; use load_lesion_arrays"):
        octlabels.load_layer_arrays(lesion_folder)


def test_volume_arrays_need_names_for_different_members(layer_folder):
    layer_set = LayerSet.load(layer_folder / "002.yml")
    layer_set.add_layer_raw("RPE", np.ones(4))
    layer_set.save(layer_folder / "002.yml")
    with pytest.raises(ValueError, match="B-scan 2 has"):
        octlabels.load_layer_arrays(layer_folder)
    arrays = octlabels.load_layer_arrays(layer_folder, ["ILM", "RPE"])
    assert arrays.confidence[:, 1, 0].tolist() == [0, 0, 1]


def test_volume_arrays_reject_different_widths(layer_folder):
    LayerSet.empty(["ILM", "BM"], 5, {"height": 20, "idx": 3}).save(
        layer_folder / "003.yml"
    )
    with pytest.raises(ValueError, match="different shapes"):
        octlabels.load_layer_arrays(layer_folder)


def test_save_arrays_single_bscan(tmp_path):
    octlabels.save_layer_arrays(
        tmp_path / "a.yml.xz", np.ones((2, 3)), names=["A", "B"]
    )
    assert octlabels.load_layer_arrays(tmp_path / "a.yml.xz").names == ["A", "B"]
    octlabels.save_lesion_arrays(tmp_path / "a.png", np.ones((1, 2, 2)), ["IRF"])
    assert octlabels.load_lesion_arrays(tmp_path / "a.png").masks.all()


def test_save_arrays_reject_wrong_dimensions(tmp_path):
    with pytest.raises(ValueError, match="2D or 3D"):
        octlabels.save_layer_arrays(tmp_path / "x.yml", np.ones(3))
    with pytest.raises(ValueError, match="3D or 4D"):
        octlabels.save_lesion_arrays(tmp_path / "x.png", np.ones((2, 2)))


def test_empty_volumes_give_empty_arrays():
    values, confidence = LayerVolume().to_array(["A"])
    assert values.shape == confidence.shape == (0, 1, 0)
    assert LayerVolume().to_array()[0].shape == (0, 0, 0)


def test_volume_type_detection_errors(layer_folder, lesion_folder, tmp_path):
    with zipfile.ZipFile(tmp_path / "both.zip", "w") as archive:
        archive.write(layer_folder / "000.yml", "layers/000.yml")
        archive.write(lesion_folder / "000.png", "lesions/000.png")
    with pytest.raises(ValueError, match="contains layer and lesion files"):
        octlabels.load(tmp_path / "both.zip")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No layer or lesion files"):
        octlabels.load_bscan(tmp_path / "empty", 0)
    with pytest.raises(NotADirectoryError):
        octlabels.load(tmp_path / "missing.zip")
