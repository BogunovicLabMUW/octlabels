import gzip
import lzma

import numpy as np
import pytest
import yaml

from octlabels import LayerSet
from octlabels.layers.serialization import dumps, loads

LEGACY = """\
info:
  width: 4
  height: 20
  idx: 0

layers:
  - ILM:
    - values: 10.5,  11,  12, nan
    - confid:    1,   1,   0,   0
    - uncrtn:    0,  40, 494, 999

regions:
  - DRUSEN:
    - values: 0,1,1,0
    - confid: 200,200,220,230
  - IRF:
    - values: 0,0,0,1
    - confid:
"""


def test_reads_legacy_format():
    ls = loads(LEGACY)
    assert ls.info == {"width": 4, "height": 20, "idx": 0}
    ilm = ls.layers["ILM"]
    np.testing.assert_array_equal(ilm.values, [10.5, 11, 12, np.nan])
    assert ilm.confidence.tolist() == [1, 1, 0, 0]
    assert ilm.uncertainty.tolist() == [0, 40, 494, 999]
    assert ls.regions["DRUSEN"].confidence.tolist() == [200, 200, 220, 230]
    assert ls.regions["IRF"].confidence is None  # empty row means "not stored"


def test_writes_legacy_layout_with_aligned_regions():
    expected = LEGACY.replace(
        "    - values: 0,1,1,0\n", "    - values:   0,  1,  1,  0\n"
    ).replace("    - confid:\n", "")
    assert dumps(loads(LEGACY)) == expected


def test_rows_are_aligned_including_custom_properties(layer_set):
    lines = [
        line for line in dumps(layer_set).splitlines() if line.startswith("    - ")
    ]
    for annotation_rows in (lines[0:4], lines[4:8], lines[8:10]):
        commas = {
            tuple(i for i, ch in enumerate(row) if ch == ",") for row in annotation_rows
        }
        assert len(commas) == 1, annotation_rows
    assert lines[3].startswith("    - snr:    ")


@pytest.mark.parametrize("yaml_arrays", [False, True])
def test_roundtrip_keeps_all_arrays(layer_set, yaml_arrays):
    loaded = loads(dumps(layer_set, yaml_arrays=yaml_arrays))
    assert loaded.info == layer_set.info
    for section, group in layer_set.sections().items():
        loaded_group = loaded.sections()[section]
        assert list(loaded_group) == list(group)
        for name, annotation in group.items():
            other = loaded_group[name]
            np.testing.assert_allclose(other.values, annotation.values)
            assert other.properties.keys() == annotation.properties.keys()
            for key, array in annotation.properties.items():
                np.testing.assert_allclose(other.properties[key], array)


def test_yaml_arrays_are_real_yaml_numbers(layer_set):
    layer_set.layers["ILM"].values[1] = np.nan
    data = yaml.safe_load(dumps(layer_set, yaml_arrays=True))
    rows = data["layers"][0]["ILM"]
    assert rows[0]["values"][0] == 5
    assert np.isnan(rows[0]["values"][1])
    assert rows[1]["confid"] == [1] * 8


def test_custom_property_dtype_is_inferred():
    ls = loads(
        "layers:\n  - A:\n    - values: 1,2\n    - grader: 3,4\n    - snr: 0.5,1\n"
    )
    assert ls.layers["A"].properties["grader"].dtype == np.int64
    assert ls.layers["A"].properties["snr"].dtype == np.float64


@pytest.mark.parametrize(
    "name, note",
    [
        ("Layer 0", "scan #3"),
        ("a: b", "yes"),
        ("1", "2024-01-01"),
        ("x", " padded "),
        ("y", "two\nlines"),
    ],
)
def test_names_and_info_strings_roundtrip(name, note):
    ls = LayerSet({"note": note})
    ls.add_layer_raw(name, [1, 2])
    loaded = loads(dumps(ls))
    assert loaded.info["note"] == note
    assert list(loaded.layers) == [name]


def test_single_value_rows():
    ls = loads("layers:\n  - A:\n    - values: 5\n    - confid: 1\n")
    assert ls.layers["A"].values.tolist() == [5.0]


def test_unknown_section_warns():
    with pytest.warns(UserWarning, match="unknown section 'extra'"):
        loads("extra: 1\nlayers:\n  - A:\n    - values: 1\n")


@pytest.mark.parametrize(
    "compress, suffix", [("none", ""), ("gzip", ".gz"), ("lzma", ".xz")]
)
def test_save_and_load_files(layer_set, tmp_path, compress, suffix):
    path = layer_set.save(tmp_path / "sub" / "000.yml", compress=compress)
    assert path.name == "000.yml" + suffix
    assert LayerSet.load(path).to_string() == layer_set.to_string()


def test_save_to_bare_filename(layer_set, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    layer_set.save("bare.yml")
    assert (tmp_path / "bare.yml").exists()


def test_load_rejects_files_without_annotations(tmp_path):
    (tmp_path / "x.yml").write_text("info:\n  width: 3\n")
    with pytest.raises(ValueError, match="no annotations"):
        LayerSet.load(tmp_path / "x.yml")


@pytest.mark.parametrize(
    "text, message",
    [
        ("- a\n- b\n", "Expected a mapping"),
        ("layers:\n  - B\n", "Expected '- NAME:'"),
        ("layers:\n  - A: 1\n", "expected a list of rows"),
        ("layers:\n  - A:\n    - 5\n", "expected '- key: row'"),
        ("layers:\n  - A:\n    - values: 1,x\n", "could not convert"),
    ],
)
def test_malformed_files_raise_clear_errors(text, message):
    with pytest.raises(ValueError, match=message):
        loads(text)


def test_unknown_compression(layer_set, tmp_path):
    with pytest.raises(ValueError, match="Unknown compression"):
        layer_set.save(tmp_path / "x.yml", compress="zip")


def test_corrupt_compressed_file_raises(tmp_path):
    (tmp_path / "000.yml.xz").write_bytes(b"not xz data")
    with pytest.raises(lzma.LZMAError):
        LayerSet.load(tmp_path / "000.yml.xz")


def test_info_value_types_roundtrip():
    info = {
        "flag": True,
        "missing": None,
        "scale": 0.5,
        "f32": np.float32(1.5),
        "i64": np.int64(3),
        "nan": float("nan"),
        "neg_inf": float("-inf"),
    }
    ls = LayerSet(info)
    ls.add_layer_raw("A", [1])
    loaded = loads(dumps(ls)).info
    assert loaded["flag"] is True and loaded["missing"] is None
    assert loaded["scale"] == 0.5 and loaded["f32"] == 1.5 and loaded["i64"] == 3
    assert np.isnan(loaded["nan"]) and loaded["neg_inf"] == float("-inf")


def test_file_without_info_section():
    ls = loads("layers:\n  - A:\n    - values: 1,2\n")
    assert ls.info == {"width": 2}  # width is taken from the annotations
    assert dumps(ls).startswith("info:\n  width: 2\n\nlayers:\n")


def test_empty_layer_set_roundtrips():
    assert dumps(LayerSet()) == "\n"
    assert not any(loads("\n").sections().values())


def test_compression_follows_the_file_name(layer_set, tmp_path):
    assert layer_set.save(tmp_path / "a.yml.gz").name == "a.yml.gz"
    assert gzip.decompress((tmp_path / "a.yml.gz").read_bytes()).startswith(b"info:")
    assert layer_set.save(tmp_path / "b.yml.xz").name == "b.yml.xz"
    assert LayerSet.load(tmp_path / "b.yml.xz").to_string() == layer_set.to_string()
    assert layer_set.save(tmp_path / "c.yml", compress="gzip").name == "c.yml.gz"
    with pytest.raises(ValueError, match="contradicts the file name"):
        layer_set.save(tmp_path / "d.yml.gz", compress="lzma")


def test_entry_without_rows_is_an_empty_annotation():
    ls = loads("layers:\n  - A:\n")
    assert len(ls.layers["A"]) == 0
