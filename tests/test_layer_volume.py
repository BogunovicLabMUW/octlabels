import json
import lzma
import zipfile

import numpy as np
import pytest

from octlabels import LayerSet, LayerVolume


def _volume(n=3):
    values = np.arange(n * 2 * 4, dtype=float).reshape(n, 2, 4)
    return LayerVolume.from_array(values, names=["A", "B"], info={"height": 50})


def test_from_array_sets_idx_and_shapes():
    volume = _volume()
    assert [ls.info["idx"] for ls in volume] == [0, 1, 2]
    assert volume.values_to_array().shape == (3, 2, 4)
    assert volume.confidence_to_array().all()
    assert volume.info == {"height": 50, "idx": 0, "width": 4}


def test_indexing():
    volume = _volume()
    assert volume[-1] is volume.layer_sets[2]
    assert isinstance(volume[1:], LayerVolume) and len(volume[1:]) == 2
    assert volume[[2, 0]].layer_sets == [volume[2], volume[0]]


def test_save_and_load_in_numeric_order(tmp_path):
    volume = LayerVolume()
    for idx in (999, 1000, 7):
        volume.add_layer_set(LayerSet.empty(["A"], 3, {"idx": idx, "height": 5}))
    volume.save(tmp_path, compress="gzip")
    (tmp_path / "notes.txt").write_text("ignored")
    loaded = LayerVolume.load(tmp_path)
    assert [ls.info["idx"] for ls in loaded] == [7, 999, 1000]


def test_from_array_updates(tmp_path):
    volume = _volume()
    volume.values_from_array(volume.values_to_array() * 0)
    assert not volume.values_to_array().any()


def test_load_from_jsonfolder(tmp_path):
    data = {
        "height": 9,
        "layers": [{"name": "ILM", "positions": [1, 2], "valid": [1, 0]}],
    }
    (tmp_path / "3.json").write_text(json.dumps(data))
    volume = LayerVolume.load_from_jsonfolder(tmp_path)
    assert volume[0].info == {"height": 9, "idx": 3, "width": 2}
    assert volume[0]["ILM"].confidence.tolist() == [1, 0]


def test_load_errors(tmp_path):
    (tmp_path / "file.txt").write_text("x")
    with pytest.raises(NotADirectoryError):
        LayerVolume.load(tmp_path / "file.txt")
    with pytest.raises(FileNotFoundError, match="No files matching"):
        LayerVolume.load(tmp_path)


def test_property_arrays_and_copy():
    values = np.ones((2, 3, 4))
    volume = LayerVolume.from_array(values, uncertainty=np.full((2, 3, 4), 5))
    assert volume.uncertainty_to_array().shape == (2, 3, 4)
    with_confidence = LayerVolume.from_array(values, confidence=np.zeros((2, 3, 4)))
    assert not with_confidence.confidence_to_array().any()
    volume.confidence_from_array(np.zeros((2, 3, 4)))
    volume.uncertainty_from_array(np.full((2, 3, 4), 9))
    assert not volume.confidence_to_array().any()
    copy = volume.copy()
    copy[0]["Layer 0"].values[0] = 99
    assert volume[0]["Layer 0"].values[0] == 1
    assert (copy.uncertainty_to_array() == 9).all()
    with pytest.raises(ValueError, match="Expected shape"):
        volume.values_from_array(np.zeros((3, 3, 4)))


def test_repr_and_empty_info():
    assert LayerVolume().info == {}
    assert repr(_volume(2)).startswith("LayerVolume(2 B-scans")


def test_layer_sets_property_is_the_list():
    volume = _volume()
    volume.layer_sets.append(volume[0].copy())
    assert len(volume) == 4
    volume.layer_sets = volume.layer_sets[:2]
    assert len(volume) == 2


@pytest.mark.parametrize(
    "compress, method",
    [
        (None, zipfile.ZIP_DEFLATED),
        ("deflate", zipfile.ZIP_DEFLATED),
        ("lzma", zipfile.ZIP_LZMA),
        ("none", zipfile.ZIP_STORED),
    ],
)
def test_zip_save_and_load(tmp_path, compress, method):
    volume = _volume()
    volume.save(tmp_path / "sub" / "vol.zip", compress=compress)
    with zipfile.ZipFile(tmp_path / "sub" / "vol.zip") as archive:
        assert archive.namelist() == ["000.yml", "001.yml", "002.yml"]
        assert {info.compress_type for info in archive.infolist()} == {method}
    loaded = LayerVolume.load(tmp_path / "sub" / "vol.zip")
    np.testing.assert_array_equal(loaded.values_to_array(), volume.values_to_array())
    assert [ls.to_string() for ls in loaded] == [ls.to_string() for ls in volume]


def test_zip_with_yaml_arrays(tmp_path):
    _volume().save(tmp_path / "vol.zip", yaml_arrays=True)
    with zipfile.ZipFile(tmp_path / "vol.zip") as archive:
        assert "values: [" in archive.read("000.yml").decode()
    assert len(LayerVolume.load(tmp_path / "vol.zip")) == 3


def test_zip_made_by_hand_with_subfolder_and_compressed_entries(tmp_path):
    volume = _volume()
    volume.save(tmp_path / "files", compress="lzma")
    with zipfile.ZipFile(tmp_path / "hand.zip", "w") as archive:
        for path in sorted((tmp_path / "files").iterdir(), reverse=True):
            archive.write(path, f"patient/layers/{path.name}")
        archive.writestr("patient/readme.txt", "ignored")
    loaded = LayerVolume.load(tmp_path / "hand.zip")
    assert [ls.info["idx"] for ls in loaded] == [0, 1, 2]
    np.testing.assert_array_equal(loaded.values_to_array(), volume.values_to_array())


def test_zip_errors(tmp_path):
    volume = _volume()
    with pytest.raises(ValueError, match="Unknown zip compression"):
        volume.save(tmp_path / "vol.zip", compress="gzip")
    with zipfile.ZipFile(tmp_path / "dup.zip", "w") as archive:
        archive.writestr("a/000.yml", volume[0].to_string())
        archive.writestr("b/000.yml.xz", lzma.compress(volume[0].to_string().encode()))
    with pytest.raises(ValueError, match="B-scan 0 is in .* twice"):
        LayerVolume.load(tmp_path / "dup.zip")
    with zipfile.ZipFile(tmp_path / "empty.zip", "w") as archive:
        archive.writestr("notes.txt", "x")
    with pytest.raises(FileNotFoundError, match="No files matching"):
        LayerVolume.load(tmp_path / "empty.zip")
    (tmp_path / "fake.zip").write_text("not a zip")
    with pytest.raises(NotADirectoryError, match="neither a folder nor a zip"):
        LayerVolume.load(tmp_path / "fake.zip")


def test_duplicate_idx_is_rejected_when_saving(tmp_path):
    volume = _volume()
    volume[2].info["idx"] = 1
    for target in (tmp_path / "folder", tmp_path / "vol.zip"):
        with pytest.raises(ValueError, match="saved as '001.yml'"):
            volume.save(target)


@pytest.mark.parametrize("target", ["folder", "vol.zip"])
def test_load_bscan_reads_only_one(tmp_path, target):
    volume = _volume()
    volume.save(tmp_path / target)
    if target == "folder":
        (tmp_path / "folder" / "000.yml").write_text("broken: [")  # never read
    bscan = LayerVolume.load_bscan(tmp_path / target, 2)
    assert bscan.info["idx"] == 2
    np.testing.assert_array_equal(bscan.values_to_array(), volume[2].values_to_array())
    with pytest.raises(KeyError, match="B-scan 7 not found"):
        LayerVolume.load_bscan(tmp_path / target, 7)
