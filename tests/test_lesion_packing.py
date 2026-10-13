import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from octlabels.lesions import packing
from octlabels.lesions.packing import (
    load_lesion_dicts,
    load_lesions,
    save_lesion_dicts,
    save_lesions,
)


def _exclusive(count, height=12, width=10):
    """`count` non-overlapping masks: mask i covers row i."""
    masks = np.zeros((count, height, width), dtype=np.uint8)
    for i in range(count):
        masks[i, i % height, : 1 + i // height] = 1
    return masks


def _overlays(count, height=12, width=10):
    masks = np.zeros((count, height, width), dtype=np.uint8)
    masks[:, :, -1] = 1  # all overlap each other and the main lesions
    return masks


def test_roundtrip_with_names(tmp_path):
    main, overlays = _exclusive(3), _overlays(2)
    save_lesions(
        tmp_path / "a.png", main, overlays, ["IRF", "SRF", "PED"], ["HRF", "X"]
    )
    m, o, mn, on = load_lesions(str(tmp_path / "a.png"))
    np.testing.assert_array_equal(m, main)
    np.testing.assert_array_equal(o, overlays)
    assert (mn, on) == (["IRF", "SRF", "PED"], ["HRF", "X"])


def test_dicts_roundtrip_and_2d_masks(tmp_path):
    main = {"IRF": _exclusive(2)[0], "SRF": _exclusive(2)[1:]}  # [H, W] and [1, H, W]
    save_lesion_dicts(tmp_path / "a.png", main, {"HRF": _overlays(1)[0]})
    m, o = load_lesion_dicts(tmp_path / "a.png")
    assert list(m) == ["IRF", "SRF"] and list(o) == ["HRF"]
    np.testing.assert_array_equal(m["SRF"], main["SRF"][0])


def test_dict_with_only_overlays_keeps_size(tmp_path):
    save_lesion_dicts(tmp_path / "a.png", {}, {"HRF": _overlays(1, 7, 5)[0]})
    m, o, _, _ = load_lesions(tmp_path / "a.png")
    assert m.shape == (0, 7, 5) and o.shape == (1, 7, 5)


@pytest.mark.parametrize("bits", range(1, 8))
def test_capacity_of_every_bit_split(tmp_path, bits):
    main, overlays = _exclusive((1 << bits) - 1, 130, 2), _overlays(8 - bits, 130, 2)
    save_lesions(tmp_path / "a.png", main, overlays, main_lesion_bit_count=bits)
    m, o, _, _ = load_lesions(tmp_path / "a.png")
    np.testing.assert_array_equal(m, main)
    np.testing.assert_array_equal(o, overlays)
    with pytest.raises(ValueError, match="Main Lesion Count"):
        save_lesions(
            tmp_path / "b.png", _exclusive(1 << bits, 260, 2), None, None, None, bits
        )
    with pytest.raises(ValueError, match="Overlay Lesion Count"):
        save_lesions(
            tmp_path / "b.png", main, _overlays(9 - bits, 130, 2), None, None, bits
        )


def test_default_load_uses_the_files_bit_split(tmp_path):
    main, overlays = _exclusive(2), _overlays(5)
    save_lesions(tmp_path / "a.png", main, overlays, main_lesion_bit_count=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m, o, _, _ = load_lesions(tmp_path / "a.png")
    np.testing.assert_array_equal(o, overlays)


@pytest.mark.parametrize("saved, loaded", [(5, 6), (6, 5)])
def test_explicit_different_bit_split_still_decodes_and_warns(tmp_path, saved, loaded):
    main, overlays = _exclusive(2), _overlays(1)
    save_lesions(tmp_path / "a.png", main, overlays, main_lesion_bit_count=saved)
    with pytest.warns(UserWarning, match=f"specifies {saved} bits"):
        m, o, _, _ = load_lesions(tmp_path / "a.png", main_lesion_bit_count=loaded)
    np.testing.assert_array_equal(m, main)
    np.testing.assert_array_equal(o, overlays)


def test_load_capacity_error_for_too_small_split(tmp_path):
    save_lesions(tmp_path / "a.png", _exclusive(6), _overlays(1))
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="Cannot load 6"):
        load_lesions(tmp_path / "a.png", main_lesion_bit_count=2)
    save_lesions(tmp_path / "b.png", _exclusive(1), _overlays(2))
    with (
        pytest.warns(UserWarning),
        pytest.raises(ValueError, match="Cannot load 2 overlay"),
    ):
        load_lesions(tmp_path / "b.png", main_lesion_bit_count=7)


def test_missing_names_get_defaults(tmp_path):
    save_lesions(tmp_path / "a.png", _exclusive(2), _overlays(2))
    m, o, mn, on = load_lesions(tmp_path / "a.png")
    assert mn is None and on is None
    md, od = load_lesion_dicts(tmp_path / "a.png")
    assert list(md) == ["L1", "L2"] and list(od) == ["Ov1", "Ov2"]


def test_empty_and_tiny_images(tmp_path):
    save_lesions(tmp_path / "e.png", np.zeros((0, 50, 40), dtype=np.uint8))
    m, o, _, _ = load_lesions(tmp_path / "e.png")
    assert m.shape == (0, 50, 40) and o.shape == (0, 50, 40)
    save_lesions(tmp_path / "t.png", np.ones((1, 1, 1), dtype=np.uint8))
    assert load_lesions(tmp_path / "t.png")[0].tolist() == [[[1]]]


def test_nonbinary_masks_count_as_present(tmp_path):
    main = _exclusive(1) * 255
    save_lesions(tmp_path / "a.png", main)
    assert load_lesions(tmp_path / "a.png")[0].max() == 1


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda p: save_lesions(p, np.zeros((5, 5))), "must be \\(C, H, W\\)"),
        (
            lambda p: save_lesions(p, _exclusive(1), np.zeros((5, 5))),
            "must be \\(K, H, W\\)",
        ),
        (
            lambda p: save_lesions(p, _exclusive(1), _overlays(1, 5, 5)),
            "Dimension mismatch",
        ),
        (
            lambda p: save_lesions(p, _exclusive(2), None, ["A"]),
            "Count mismatch for Main",
        ),
        (
            lambda p: save_lesions(p, _exclusive(1), _overlays(1), None, ["A", "B"]),
            "Count mismatch for Overlay",
        ),
        (lambda p: save_lesions(p, np.ones((2, 4, 4))), "mutually exclusive"),
        (
            lambda p: save_lesions(p, _exclusive(1), main_lesion_bit_count=8),
            "between 1",
        ),
    ],
)
def test_save_validation(tmp_path, call, message):
    with pytest.raises(ValueError, match=message):
        call(tmp_path / "a.png")


def _write_png(path: Path, text=None, mode="P"):
    info = PngImagePlugin.PngInfo()
    if text is not None:
        info.add_text(packing.METADATA_KEY, text, zip=True)
    Image.new(mode, (4, 3)).save(path, pnginfo=info)


def test_metadata_errors(tmp_path):
    _write_png(tmp_path / "none.png")
    with pytest.raises(ValueError, match="missing 'Lesions' metadata"):
        load_lesions(tmp_path / "none.png")
    _write_png(tmp_path / "bad.png", "{not json")
    with pytest.raises(ValueError, match="corrupted"):
        load_lesions(tmp_path / "bad.png")
    _write_png(tmp_path / "nocount.png", json.dumps({"v": "1"}))
    with pytest.raises(ValueError, match="could not be determined"):
        load_lesions(tmp_path / "nocount.png")


def test_metadata_without_bit_count_uses_default(tmp_path):
    _write_png(tmp_path / "a.png", json.dumps({"mc": 1, "oc": 0}))
    m, o, _, _ = load_lesions(tmp_path / "a.png")
    assert m.shape == (1, 3, 4)


def test_count_override_warns(tmp_path):
    save_lesions(tmp_path / "a.png", _exclusive(3))
    with pytest.warns(UserWarning, match="differs from provided argument"):
        m, _, _, _ = load_lesions(tmp_path / "a.png", main_lesion_count=5)
    assert m.shape[0] == 5 and not m[3:].any()


def test_non_palette_image_warns(tmp_path):
    _write_png(tmp_path / "rgb.png", json.dumps({"bc": 6, "mc": 0, "oc": 0}), "RGB")
    with pytest.warns(UserWarning, match="expected 'P'"), pytest.raises(ValueError):
        load_lesions(tmp_path / "rgb.png")


def test_palette_colors():
    palette = packing._palette(6, 63 + 64)
    colors = [tuple(palette[i * 3 : i * 3 + 3]) for i in range(len(palette) // 3)]
    assert colors[0] == (0, 0, 0)
    assert len(set(colors[1:17])) == 16  # fixed, distinct main colors
    assert colors[20] == tuple(packing._palette(6, 20)[60:63])  # deterministic
    # Index 64 = no main lesion + second overlay bit: gray blended with cyan.
    assert colors[64] == (64, 191, 191)
    # Index 65 = main lesion 1 (red) + the same overlay: tinted red.
    assert colors[65] == (127, 127, 127)


def test_matching_explicit_arguments_do_not_warn(tmp_path):
    save_lesions(tmp_path / "a.png", _exclusive(2), _overlays(1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m, o, _, _ = load_lesions(
            tmp_path / "a.png",
            main_lesion_bit_count=6,
            main_lesion_count=2,
            overlay_lesion_count=1,
        )
    assert m.shape[0] == 2 and o.shape[0] == 1


def test_empty_dicts_give_a_1x1_image(tmp_path):
    save_lesion_dicts(tmp_path / "a.png", {})
    m, o, _, _ = load_lesions(tmp_path / "a.png")
    assert m.shape == o.shape == (0, 1, 1)
