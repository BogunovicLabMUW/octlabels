import numpy as np
import pytest

from octlabels import Lesion, LesionSet, LesionVolume


@pytest.fixture
def lesion_set():
    masks = np.zeros((3, 6, 8), dtype=np.uint8)
    masks[0, 1] = 1
    masks[1, 3] = 1
    masks[2, :, 0] = 1  # overlay crossing both
    return LesionSet.from_array(
        masks, ["IRF", "SRF", "HRF"], {"idx": 4}, main_lesion_names=["IRF", "SRF"]
    )


def test_lesion_mask_is_binary_2d():
    lesion = Lesion("IRF", [[0, 255], [3, 0]])
    assert lesion.mask.tolist() == [[0, 1], [1, 0]] and lesion.mask.dtype == np.uint8
    with pytest.raises(ValueError, match="must be 2D"):
        Lesion("IRF", np.zeros(4))
    with pytest.raises(ValueError, match="expected \\(2, 2\\)"):
        lesion.mask = np.zeros((3, 3))
    lesion.mask = np.ones((2, 2))
    assert lesion.mask.all()
    assert (
        repr(Lesion("HRF", np.zeros((2, 3))))
        == "Lesion(name='HRF', overlay, shape=(2, 3))"
    )


def test_add_lesion_object(lesion_set):
    lesion_set.add_lesion(Lesion("PED", np.zeros((6, 8)), main=True))
    assert lesion_set.main_lesion_names == ["IRF", "SRF", "PED"]
    with pytest.raises(ValueError, match="already exists"):
        lesion_set.add_lesion(Lesion("PED", np.zeros((6, 8))))


def test_shape_is_claimed_and_enforced(lesion_set):
    assert lesion_set.shape == (3, 6, 8)
    assert lesion_set.info == {"idx": 4, "height": 6, "width": 8}
    with pytest.raises(ValueError, match="has height 5"):
        lesion_set.add_lesion_raw("PED", np.zeros((5, 8)))


def test_main_lesion_names(lesion_set):
    assert lesion_set.main_lesion_names == ["IRF", "SRF"]
    lesion_set.main_lesion_names = ["SRF"]
    assert [lesion.main for lesion in lesion_set.lesions.values()] == [
        False,
        True,
        False,
    ]
    with pytest.raises(KeyError):
        lesion_set.main_lesion_names = ["PED"]


def test_access(lesion_set):
    assert lesion_set[0] is lesion_set["IRF"] is lesion_set.lesions["IRF"]
    assert "HRF" in lesion_set and len(lesion_set) == 3
    with pytest.raises(TypeError):
        lesion_set[1.5]
    with pytest.raises(KeyError, match="Lesion 'PED' not found"):
        lesion_set["PED"]
    assert repr(lesion_set).startswith("LesionSet(info=")


def test_copy_subset_ensure_and_rename_keep_main_flags(lesion_set):
    copy = lesion_set.copy()
    copy["IRF"].mask[:] = 0
    assert lesion_set["IRF"].mask.any()
    assert copy.main_lesion_names == ["IRF", "SRF"]
    assert lesion_set.subset(["HRF", "SRF"]).main_lesion_names == ["SRF"]
    ensured = lesion_set.ensure(["SRF", "PED"])
    assert list(ensured.lesions) == ["SRF", "PED"]
    assert ensured.main_lesion_names == ["SRF"] and not ensured["PED"].mask.any()
    with pytest.raises(KeyError):
        lesion_set.subset(["PED"])
    lesion_set.lesions.rename({"IRF": "Cyst"})
    assert lesion_set.main_lesion_names == ["Cyst", "SRF"]


def test_to_array(lesion_set):
    assert lesion_set.to_array().shape == (3, 6, 8)
    assert lesion_set.to_array(["HRF"])[0, :, 0].all()
    assert LesionSet({"height": 2, "width": 3}).to_array().shape == (0, 2, 3)


def test_from_array_validation():
    with pytest.raises(ValueError, match="\\(C, H, W\\)"):
        LesionSet.from_array(np.zeros((4, 4)), ["A"])
    with pytest.raises(ValueError, match="1 names for 2 masks"):
        LesionSet.from_array(np.zeros((2, 4, 4)), ["A"])


def test_save_load_keeps_grouping_main_first(lesion_set, tmp_path):
    lesion_set.main_lesion_names = ["SRF"]
    lesion_set.save(tmp_path / "a.png")
    loaded = LesionSet.load(tmp_path / "a.png")
    assert list(loaded.lesions) == ["SRF", "IRF", "HRF"]  # main lesions first
    assert loaded.main_lesion_names == ["SRF"]
    np.testing.assert_array_equal(loaded["IRF"].mask, lesion_set["IRF"].mask)
    loaded.save(tmp_path / "b.png")
    assert LesionSet.load(tmp_path / "b.png").main_lesion_names == ["SRF"]


def test_save_with_explicit_grouping_and_bits(lesion_set, tmp_path):
    lesion_set.save(
        tmp_path / "a.png", main_lesion_names=["IRF"], main_lesion_bit_count=3
    )
    loaded = LesionSet.load(tmp_path / "a.png")
    assert loaded.main_lesion_names == ["IRF"]
    with pytest.warns(UserWarning, match="specifies 3 bits"):
        LesionSet.load(tmp_path / "a.png", main_lesion_bit_count=4)


def test_empty_file_keeps_size(tmp_path):
    LesionSet({"height": 5, "width": 7}).save(tmp_path / "e.png")
    loaded = LesionSet.load(tmp_path / "e.png")
    assert loaded.shape == (0, 5, 7)


def test_default_names_for_unnamed_files(tmp_path):
    from octlabels.lesions.packing import save_lesions

    masks = np.zeros((1, 2, 2), dtype=np.uint8)
    save_lesions(tmp_path / "a.png", masks, masks)
    assert list(LesionSet.load(tmp_path / "a.png").lesions) == ["L1", "Ov1"]


def _volume():
    masks = np.zeros((3, 2, 4, 5), dtype=np.uint8)
    masks[:, 0, 1] = 1
    masks[1, 1, :, 0] = 1
    return LesionVolume.from_array(masks, ["SRF", "HRF"], {"source": "manual"}, ["SRF"])


def test_volume_arrays_and_indexing():
    volume = _volume()
    assert volume.to_array().shape == (3, 2, 4, 5)
    assert volume.to_array(["HRF", "PED"])[1, 0, :, 0].all()
    assert not volume.to_array(["PED"]).any()
    assert [ls.info["idx"] for ls in volume[1:]] == [1, 2]
    assert isinstance(volume[[2, 0]], LesionVolume)
    assert volume.info == {"source": "manual", "idx": 0, "height": 4, "width": 5}
    assert LesionVolume().to_array(["A"]).shape == (0, 1, 0, 0)
    with pytest.raises(ValueError, match="\\(B, C, H, W\\)"):
        LesionVolume.from_array(np.zeros((2, 4, 5)), ["A"])


def test_volume_sets_property():
    volume = _volume()
    volume.add_lesion_set(volume[0].copy())
    assert len(volume.lesion_sets) == 4
    volume.lesion_sets = volume.lesion_sets[:1]
    assert len(volume) == 1


def test_volume_save_load(tmp_path):
    volume = _volume()
    volume[2].info["idx"] = 1000
    volume.save(tmp_path)
    (tmp_path / "notes.txt").write_text("ignored")
    loaded = LesionVolume.load(tmp_path)
    assert [ls.info["idx"] for ls in loaded] == [0, 1, 1000]  # from the file names
    assert loaded[1].main_lesion_names == ["SRF"]
    np.testing.assert_array_equal(loaded.to_array(), volume.to_array())
    copy = loaded.copy()
    copy[0]["SRF"].mask[:] = 0
    assert loaded[0]["SRF"].mask.any()
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No files matching"):
        LesionVolume.load(tmp_path / "empty")


@pytest.mark.parametrize("compress", [None, "lzma", "none"])
def test_lesion_volume_zip(tmp_path, compress):
    volume = _volume()
    volume.save(tmp_path / "vol.zip", compress=compress)
    loaded = LesionVolume.load(tmp_path / "vol.zip")
    np.testing.assert_array_equal(loaded.to_array(), volume.to_array())
    assert loaded[1].main_lesion_names == ["SRF"]
    bscan = LesionVolume.load_bscan(tmp_path / "vol.zip", 1)
    np.testing.assert_array_equal(bscan.to_array(), volume[1].to_array())


def test_lesion_volume_load_bscan_from_folder(tmp_path):
    _volume().save(tmp_path)
    (tmp_path / "000.png").write_bytes(b"broken")  # never read
    assert LesionVolume.load_bscan(tmp_path, 2).info["idx"] == 2
    with pytest.raises(KeyError, match="B-scan 5 not found"):
        LesionVolume.load_bscan(tmp_path, 5)


def test_lesion_volume_compress_only_for_zip(tmp_path):
    with pytest.raises(ValueError, match="only applies to zip"):
        _volume().save(tmp_path / "folder", compress="lzma")


def test_lesion_volume_to_array_requires_same_lesions_without_names():
    volume = _volume()
    volume[2].add_lesion_raw("PED", np.zeros((4, 5)))
    with pytest.raises(ValueError, match="B-scan 2 has"):
        volume.to_array()
    assert volume.to_array(["SRF", "PED"]).shape == (3, 2, 4, 5)


def test_lesion_set_to_array_fills_missing_with_empty_masks(lesion_set):
    masks = lesion_set.to_array(["SRF", "PED"])
    assert masks.shape == (2, 6, 8) and not masks[1].any()


def test_lesion_set_from_array_default_names():
    lesion_set = LesionSet.from_array(np.zeros((2, 3, 3)))
    assert list(lesion_set.lesions) == ["Lesion 0", "Lesion 1"]


def test_save_creates_parent_folders(lesion_set, tmp_path):
    lesion_set.save(tmp_path / "new" / "folder" / "000.png")
    assert LesionSet.load(tmp_path / "new" / "folder" / "000.png").shape == (3, 6, 8)
