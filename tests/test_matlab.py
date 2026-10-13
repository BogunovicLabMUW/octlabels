import os
import subprocess
import sys

import numpy as np
import pytest

import octlabels
from octlabels.layers import matlab

scipy_io = pytest.importorskip("scipy.io")


def _duke_dme_like(num_bscans=4, width=5):
    """[8 boundaries, A-scans, B-scans], only B-scans 1 and 3 annotated."""
    layers = np.full((8, width, num_bscans), np.nan)
    for b in (1, 3):
        layers[:, :, b] = np.arange(8)[:, None] * 10 + 1 + b
    layers[2, 0, 1] = np.nan  # one missing point
    return {"manualLayers1": layers, "images": np.zeros((100, width, num_bscans))}


def test_duke_dme_layout(tmp_path):
    path = tmp_path / "Subject_01.mat"
    scipy_io.savemat(path, _duke_dme_like())
    volume = matlab.load_duke_mat(path)
    assert [ls.info["idx"] for ls in volume] == [1, 3]
    bscan = volume[0]
    assert bscan.info == {"width": 5, "height": 100, "idx": 1}
    assert list(bscan.layers) == list(matlab.DUKE_VARIABLES["manualLayers1"][1])
    assert bscan["ILM"].values.tolist() == [1] * 5  # 1-based 2 -> 0-based 1
    assert bscan["IPL-INL"].confidence.tolist() == [0, 1, 1, 1, 1]
    assert bscan["IPL-INL"].values[0] == 0
    bscan.validate()


def test_keep_unannotated_bscans(tmp_path):
    path = tmp_path / "s.mat"
    scipy_io.savemat(path, _duke_dme_like())
    volume = matlab.load_duke_mat(path, skip_unannotated=False, one_based=False)
    assert len(volume) == 4
    assert not volume[0].confidence_to_array().any()
    assert volume[1]["ILM"].values[0] == 2


def test_duke_amd_layout(tmp_path):
    layer_maps = np.random.default_rng(0).uniform(10, 40, size=(3, 6, 3))  # B, A, L
    path = tmp_path / "AMD_Subject_1001.mat"
    scipy_io.savemat(path, {"layerMaps": layer_maps, "Age": 70})
    volume = matlab.load_duke_mat(path, height=64)
    assert len(volume) == 3
    assert list(volume[0].layers) == ["ILM", "RPEDC", "BM"]
    np.testing.assert_allclose(
        volume[2]["BM"].values, layer_maps[2, :, 2] - 1, rtol=1e-6
    )


def test_custom_variable_needs_axes(tmp_path):
    path = tmp_path / "custom.mat"
    scipy_io.savemat(path, {"boundaries": np.ones((2, 3, 4))})  # B, L, A
    with pytest.raises(KeyError, match="Available: \\['boundaries'\\]"):
        matlab.load_duke_mat(path)
    with pytest.raises(KeyError, match="'missing' not in"):
        matlab.load_duke_mat(path, variable="missing")
    with pytest.raises(ValueError, match="permutation of 'LAB'"):
        matlab.load_duke_mat(path, variable="boundaries")
    volume = matlab.load_duke_mat(
        path, variable="boundaries", axes="BLA", names=["A", "B", "C"]
    )
    assert len(volume) == 2 and volume[0].width == 4
    assert list(volume[0].layers) == ["A", "B", "C"]
    unnamed = matlab.load_duke_mat(path, variable="boundaries", axes="BLA")
    assert list(unnamed[0].layers) == ["Layer 0", "Layer 1", "Layer 2"]


def test_matlab_v73_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    data = _duke_dme_like()
    path = tmp_path / "v73.mat"
    with h5py.File(path, "w", userblock_size=512) as file:
        for name, array in data.items():
            file[name] = array.T  # MATLAB writes reversed dimensions
    with open(path, "r+b") as file:
        file.write(b"MATLAB 7.3 MAT-file, Platform: GLNXA64, HDF5 schema 1.00 .")
    volume = matlab.load_duke_mat(path)
    assert [ls.info["idx"] for ls in volume] == [1, 3]
    assert volume[0].height == 100
    assert volume[0]["ILM"].values.tolist() == [1] * 5


@pytest.mark.parametrize("module, header", [("scipy.io", b""), ("h5py", b"MATLAB 7.3")])
def test_missing_optional_package_gives_install_hint(
    tmp_path, monkeypatch, module, header
):
    path = tmp_path / "x.mat"
    path.write_bytes(header + b" " * 200)
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match='pip install "octlabels\\[matlab\\]"'):
        matlab.load_duke_mat(path)


def test_importing_octlabels_does_not_import_optional_packages():
    code = (
        "import sys, octlabels; "
        "print(octlabels.__file__); "
        "print('scipy' in sys.modules or 'h5py' in sys.modules)"
    )
    # The child process must import this checkout, not an installed copy.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    imported_from, optional_loaded = result.stdout.split()
    assert imported_from == octlabels.__file__
    assert optional_loaded == "False"
