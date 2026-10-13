"""Checks real data files. Skipped unless OCTLABELS_TESTCASES points to a folder;
every ``NNN.yml[.gz|.xz]`` below it and every folder containing them is tested.
"""

import gzip
import lzma
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from octlabels import LayerSet, LayerVolume, LesionSet, LesionVolume
from octlabels.layers import rasterize, transforms
from octlabels.layers.serialization import dumps, loads

ROOT = os.environ.get("OCTLABELS_TESTCASES")
FILE = re.compile(r"^\d+\.yml(\.gz|\.xz)?$")
LESION_FILE = re.compile(r"^\d+\.png$")
ALL = sorted(Path(ROOT).rglob("*")) if ROOT else []
FILES = [p for p in ALL if FILE.match(p.name)]
FOLDERS = sorted({p.parent for p in FILES})
LESION_FILES = [p for p in ALL if LESION_FILE.match(p.name)]
LESION_FOLDERS = sorted({p.parent for p in LESION_FILES})

pytestmark = pytest.mark.skipif(not ROOT, reason="OCTLABELS_TESTCASES not set")


def _id(path: Path) -> str:
    return str(path.relative_to(ROOT))[-60:]


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if path.name.endswith(".gz"):
        data = gzip.decompress(data)
    elif path.name.endswith(".xz"):
        data = lzma.decompress(data)
    return data.decode("utf-8")


def _unpadded(text: str) -> str:
    return re.sub(r"(?<=[:,]) +", "", text)


@pytest.mark.parametrize("path", FILES, ids=_id)
def test_file_roundtrips(path):
    """Saving reproduces the file up to column padding (older writers padded
    to the width of ``values`` only), and is stable in both row styles."""
    text = _read_text(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layer_set = loads(text)
    assert any(layer_set.sections().values())
    written = dumps(layer_set)
    assert _unpadded(written) == _unpadded(text)
    assert dumps(loads(written)) == written
    assert dumps(loads(dumps(layer_set, yaml_arrays=True))) == written


@pytest.mark.parametrize("path", FILES, ids=_id)
def test_file_is_valid_and_transformable(path):
    layer_set = LayerSet.load(path)
    layer_set.validate()
    assert int(path.name.split(".")[0]) == layer_set.info.get("idx", -1)
    width = layer_set.width
    for result in (
        transforms.flip(layer_set),
        transforms.resize_horizontal(layer_set, width // 2),
        transforms.rotate(layer_set, np.deg2rad(5)),
        layer_set.ensure(list(layer_set.layers)[:3] + ["MISSING"]),
    ):
        for group in result.sections().values():
            for annotation in group.values():
                for array in (annotation.values, *annotation.properties.values()):
                    assert array.shape == (result.width,)
    masks, _ = rasterize.generate_2d_segmentation(layer_set)
    assert masks.shape == (len(layer_set.layers) + 1, layer_set.height, width)


@pytest.mark.parametrize("folder", FOLDERS, ids=_id)
def test_folder_roundtrips_through_zip(folder, tmp_path):
    volume = LayerVolume.load(folder)
    volume.save(tmp_path / "vol.zip", compress="lzma")
    loaded = LayerVolume.load(tmp_path / "vol.zip")
    assert [ls.to_string() for ls in loaded] == [ls.to_string() for ls in volume]
    middle = volume[len(volume) // 2]
    single = LayerVolume.load_bscan(tmp_path / "vol.zip", middle.info["idx"])
    assert single.to_string() == middle.to_string()


@pytest.mark.parametrize("folder", FOLDERS, ids=_id)
def test_folder_loads_as_volume(folder):
    volume = LayerVolume.load(folder)
    assert len(volume) == sum(1 for p in folder.iterdir() if FILE.match(p.name))
    assert [ls.info["idx"] for ls in volume] == sorted(ls.info["idx"] for ls in volume)
    assert volume.values_to_array().shape[0] == len(volume)


@pytest.mark.parametrize("path", LESION_FILES, ids=_id)
def test_lesion_file_roundtrips_byte_identical(path, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lesion_set = LesionSet.load(path)
    lesion_set.save(tmp_path / "out.png")
    assert (tmp_path / "out.png").read_bytes() == path.read_bytes()


@pytest.mark.parametrize("folder", LESION_FOLDERS, ids=_id)
def test_lesion_folder_loads_as_volume(folder):
    volume = LesionVolume.load(folder)
    assert len(volume) == sum(1 for p in folder.iterdir() if LESION_FILE.match(p.name))
    masks = volume.to_array()
    assert masks.shape[:2] == (len(volume), len(volume[0].lesions))
    assert set(np.unique(masks)) <= {0, 1}


@pytest.mark.parametrize("folder", LESION_FOLDERS, ids=_id)
def test_lesion_folder_roundtrips_through_zip(folder, tmp_path):
    volume = LesionVolume.load(folder)
    volume.save(tmp_path / "lesions.zip")
    np.testing.assert_array_equal(
        LesionVolume.load(tmp_path / "lesions.zip").to_array(), volume.to_array()
    )
    middle = volume[len(volume) // 2]
    single = LesionVolume.load_bscan(tmp_path / "lesions.zip", middle.info["idx"])
    np.testing.assert_array_equal(single.to_array(), middle.to_array())
