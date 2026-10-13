"""Packing of lesion masks into one 8-bit indexed PNG per B-scan.

Main lesions (mutually exclusive, e.g. fluids) are stored as an integer ID in
the low bits of each pixel; overlay lesions (may overlap anything, e.g. HRF)
are stored as one flag bit each, from the highest bit down. With the default
split of 6 bits, a file holds up to 63 main lesions and 2 overlays.

Names and the bit split are stored as compressed JSON in the PNG (``zTXt``
chunk ``"Lesions"``), and a palette makes the files viewable in any image
viewer.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, PngImagePlugin

FORMAT_VERSION: str = "1"
"""Format version written to the metadata."""
METADATA_KEY: str = "Lesions"
"""Key of the compressed JSON text chunk."""
DEFAULT_MAIN_BITS: int = 6
"""Bits for main lesions when saving, unless given."""

_MAIN_COLORS: List[Tuple[int, int, int]] = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (255, 128, 0), (128, 0, 255),
    (255, 0, 128), (0, 128, 255), (128, 255, 0), (255, 128, 128),
    (128, 128, 255), (0, 128, 128), (128, 0, 0), (0, 100, 0),
]  # fmt: skip
# Tint per overlay, blended into the main lesion color.
_OVERLAY_TINTS: List[Tuple[int, int, int]] = [
    (255, 255, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
]  # fmt: skip

FileLike = Union[str, Path, BinaryIO]
"""A file path, or an open binary file such as ``io.BytesIO``."""


def _file(target: FileLike) -> Union[Path, BinaryIO]:
    """A path as Path; an open file as it is."""
    if isinstance(target, (str, Path)):
        return Path(target)
    return target


def _overlay_bit(k: int) -> int:
    """The pixel bit of overlay ``k``: overlays use the highest bits first."""
    return 7 - k


def _palette(main_bits: int, max_index: int) -> List[int]:
    """Flat RGB palette for pixel values 0 to ``max_index``.

    Each main lesion gets its own color; set overlay bits blend a tint into it.
    """
    main_id_mask = (1 << main_bits) - 1
    num_overlays = 8 - main_bits
    palette = [0, 0, 0]  # pixel value 0: background
    for pixel in range(1, max_index + 1):
        main_id = pixel & main_id_mask
        if main_id == 0:
            color = np.array([128, 128, 128])
        elif main_id <= len(_MAIN_COLORS):
            color = np.array(_MAIN_COLORS[main_id - 1])
        else:
            color = np.random.default_rng(main_id * 100).integers(50, 255, size=3)

        tints = []
        for k in range(num_overlays):
            if pixel & (1 << _overlay_bit(k)):
                tints.append(_OVERLAY_TINTS[k % len(_OVERLAY_TINTS)])
        if tints:
            tint = np.mean(tints, axis=0).astype(int)
            color = (color.astype(int) + tint) // 2
        palette.extend(color.tolist())
    return palette


def _check_names(array: np.ndarray, names: Optional[List[str]], kind: str) -> None:
    if names is not None and len(names) != array.shape[0]:
        raise ValueError(
            f"Count mismatch for {kind}: Array has {array.shape[0]} channels, "
            f"but {len(names)} names provided."
        )


def _check_bits(main_bits: int) -> None:
    if not 1 <= main_bits <= 7:
        raise ValueError(
            f"main_lesion_bit_count must be between 1 and 7. Got {main_bits}."
        )


def save_lesions(
    filename: FileLike,
    main_arr: np.ndarray,
    overlay_arr: Optional[np.ndarray] = None,
    main_names: Optional[List[str]] = None,
    overlay_names: Optional[List[str]] = None,
    main_lesion_bit_count: int = DEFAULT_MAIN_BITS,
) -> None:
    """Saves main and overlay lesion masks into one PNG.

    Args:
        filename: Output path.
        main_arr: Mutually exclusive masks, nonzero = present. Shape: [N, H, W].
        overlay_arr: Masks that may overlap anything. Shape: [K, H, W].
        main_names: N names (stored in the file).
        overlay_names: K names.
        main_lesion_bit_count: Bits for main lesions (1-7); the remaining
            bits hold one overlay each.

    Raises:
        ValueError: On wrong shapes or name counts, overlapping main lesions,
            or more lesions than the bit split can hold.
    """
    _check_bits(main_lesion_bit_count)
    if main_arr.ndim != 3:
        raise ValueError(f"main_arr must be (C, H, W). Got shape {main_arr.shape}.")
    num_main, height, width = main_arr.shape
    if overlay_arr is not None:
        if overlay_arr.ndim != 3:
            raise ValueError(
                f"overlay_arr must be (K, H, W). Got shape {overlay_arr.shape}."
            )
        if overlay_arr.shape[1:] != (height, width):
            raise ValueError(
                f"Dimension mismatch. Main is {height}x{width}, "
                f"Overlay is {overlay_arr.shape[1]}x{overlay_arr.shape[2]}."
            )
        _check_names(overlay_arr, overlay_names, "Overlay")
    _check_names(main_arr, main_names, "Main")
    if num_main > 1 and np.sum(main_arr > 0, axis=0).max() > 1:
        raise ValueError(
            "Main Lesions must be mutually exclusive. "
            "Found pixels where multiple main lesion masks overlap."
        )

    max_main = (1 << main_lesion_bit_count) - 1
    if num_main > max_main:
        raise ValueError(
            f"Main Lesion Count ({num_main}) exceeds capacity of "
            f"{main_lesion_bit_count} bits (Max IDs: {max_main})."
        )
    num_overlays = 0 if overlay_arr is None else overlay_arr.shape[0]
    if num_overlays > 8 - main_lesion_bit_count:
        raise ValueError(
            f"Overlay Lesion Count ({num_overlays}) exceeds available bits. "
            f"Main uses {main_lesion_bit_count} bits, leaving "
            f"{8 - main_lesion_bit_count} for overlays."
        )

    packed = np.zeros((height, width), dtype=np.uint8)
    for i in range(num_main):
        packed[main_arr[i] > 0] = i + 1  # main lesion IDs start at 1
    for k in range(num_overlays):
        packed[overlay_arr[k] > 0] |= 1 << _overlay_bit(k)

    # Short keys: v=version, bc=main bits, mc/oc=counts, mn/on=names.
    meta = {
        "v": FORMAT_VERSION,
        "bc": main_lesion_bit_count,
        "mc": num_main,
        "oc": num_overlays,
    }
    if main_names:
        meta["mn"] = main_names
    if overlay_names:
        meta["on"] = overlay_names
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text(METADATA_KEY, json.dumps(meta, separators=(",", ":")), zip=True)

    image = Image.frombytes("P", (width, height), packed.tobytes())
    image.putpalette(_palette(main_lesion_bit_count, int(packed.max())))
    image.save(
        _file(filename), format="PNG", pnginfo=png_info, optimize=True, compress_level=9
    )


def _stack_dict(masks: Dict[str, np.ndarray]) -> np.ndarray:
    """Stacks the masks of a dictionary; values may be [H, W] or [1, H, W]."""
    stacked = []
    for mask in masks.values():
        if mask.ndim == 2:
            mask = mask[np.newaxis]
        stacked.append(mask)
    return np.concatenate(stacked, axis=0)


def save_lesion_dicts(
    filename: FileLike,
    main_dict: Dict[str, np.ndarray],
    overlay_dict: Optional[Dict[str, np.ndarray]] = None,
    main_lesion_bit_count: int = DEFAULT_MAIN_BITS,
) -> None:
    """Like [`save_lesions`][], with ``{name: mask}`` dictionaries."""
    overlay_arr = None
    overlay_names = None
    if overlay_dict:
        overlay_arr = _stack_dict(overlay_dict)
        overlay_names = list(overlay_dict)

    if main_dict:
        main_arr = _stack_dict(main_dict)
    elif overlay_arr is not None:
        main_arr = np.zeros((0, *overlay_arr.shape[1:]), dtype=np.uint8)
    else:
        main_arr = np.zeros((0, 1, 1), dtype=np.uint8)

    save_lesions(
        filename,
        main_arr,
        overlay_arr,
        list(main_dict),
        overlay_names,
        main_lesion_bit_count,
    )


def _resolve_count(kind: str, stored: Optional[int], given: Optional[int]) -> int:
    """An explicit count wins over the stored one, with a warning on conflict."""
    if given is not None:
        if stored is not None and given != stored:
            warnings.warn(
                f"{kind}: Metadata count ({stored}) differs from provided argument "
                f"({given}). Using explicit argument.",
                stacklevel=3,
            )
        return given
    if stored is not None:
        return stored
    raise ValueError(
        f"{kind} count could not be determined. Metadata missing and no argument "
        "provided."
    )


def load_lesions(
    filename: FileLike,
    main_lesion_bit_count: Optional[int] = None,
    main_lesion_count: Optional[int] = None,
    overlay_lesion_count: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]], Optional[List[str]]]:
    """Loads lesion masks from a PNG written by [`save_lesions`][].

    Args:
        filename: Path to the PNG.
        main_lesion_bit_count: Bit split to decode with. Defaults to the value
            stored in the file. A different value still decodes correctly as
            long as the IDs and overlays fit (main IDs are in the low bits,
            overlays in the high bits), but warns.
        main_lesion_count: Overrides the stored number of main lesions.
        overlay_lesion_count: Overrides the stored number of overlays.

    Returns:
        ``(main, overlays, main_names, overlay_names)``: uint8 masks of shape
        [N, H, W] and [K, H, W] (with N or K = 0 if there are none), and the
        stored names (None if the file has none).

    Raises:
        ValueError: If the metadata is missing or corrupt, or the lesions do not
            fit the requested bit split.
    """
    with Image.open(_file(filename)) as image:
        if image.mode not in ("P", "L"):
            warnings.warn(
                f"Image mode is '{image.mode}', expected 'P' (Palette). "
                "Loading may fail.",
                stacklevel=2,
            )
        packed = np.array(image)
        text = dict(image.text)
    height, width = packed.shape

    if METADATA_KEY not in text:
        raise ValueError(
            f"File {filename} is missing '{METADATA_KEY}' metadata. "
            "Legacy files are not supported."
        )
    try:
        meta = json.loads(text[METADATA_KEY])
    except json.JSONDecodeError:
        raise ValueError("Metadata JSON is corrupted.") from None

    stored_bits = meta.get("bc")
    if main_lesion_bit_count is None:
        if stored_bits is None:
            main_lesion_bit_count = DEFAULT_MAIN_BITS
        else:
            main_lesion_bit_count = stored_bits
    elif stored_bits is not None and stored_bits != main_lesion_bit_count:
        warnings.warn(
            f"File metadata specifies {stored_bits} bits, "
            f"but loading with {main_lesion_bit_count} bits.",
            stacklevel=2,
        )
    _check_bits(main_lesion_bit_count)
    num_main = _resolve_count("Main Lesion", meta.get("mc"), main_lesion_count)
    num_overlays = _resolve_count(
        "Overlay Lesion", meta.get("oc"), overlay_lesion_count
    )

    max_main = (1 << main_lesion_bit_count) - 1
    if num_main > max_main:
        raise ValueError(
            f"Cannot load {num_main} main lesions using {main_lesion_bit_count} "
            f"bits. Max capacity is {max_main}."
        )
    if num_overlays > 8 - main_lesion_bit_count:
        raise ValueError(
            f"Cannot load {num_overlays} overlay lesions using "
            f"{main_lesion_bit_count} main bits. Only "
            f"{8 - main_lesion_bit_count} bits remain for overlays."
        )

    main_ids = packed & max_main  # the low bits hold the main lesion ID
    main = np.zeros((num_main, height, width), dtype=np.uint8)
    for i in range(num_main):
        main[i] = main_ids == i + 1
    overlays = np.zeros((num_overlays, height, width), dtype=np.uint8)
    for k in range(num_overlays):
        overlays[k] = (packed >> _overlay_bit(k)) & 1
    return main, overlays, meta.get("mn"), meta.get("on")


def load_lesion_dicts(
    filename: FileLike,
    main_lesion_bit_count: Optional[int] = None,
    main_lesion_count: Optional[int] = None,
    overlay_lesion_count: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Like [`load_lesions`][], returning ``{name: mask}`` dictionaries.
    Missing names become ``L1, L2, ...`` and ``Ov1, Ov2, ...``."""
    main, overlays, main_names, overlay_names = load_lesions(
        filename, main_lesion_bit_count, main_lesion_count, overlay_lesion_count
    )
    main_names = names_or_default(main_names, len(main), "L")
    overlay_names = names_or_default(overlay_names, len(overlays), "Ov")
    main_dict = dict(zip(main_names, main, strict=True))
    overlay_dict = dict(zip(overlay_names, overlays, strict=True))
    return main_dict, overlay_dict


def names_or_default(names: Optional[List[str]], count: int, prefix: str) -> List[str]:
    """The stored names, or ``<prefix>1, <prefix>2, ...`` if they are missing
    or do not match the number of masks."""
    if not names or len(names) != count:
        return [f"{prefix}{i + 1}" for i in range(count)]
    return list(names)
