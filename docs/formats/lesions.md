# Lesion files

A lesion file holds the 2D lesion masks of one B-scan, packed into a single
8-bit indexed PNG named `NNN.png`.

## Main lesions and overlays

Lesions are split into two groups:

- **Main lesions** are mutually exclusive: a pixel belongs to at most one of
  them. Typical examples are fluid compartments (IRF, SRF, PED).
- **Overlays** may overlap anything, including each other. A typical example
  is hyperreflective foci (HRF).

## Pixel values

Each pixel is one byte. With a split of `b` bits for main lesions (6 by
default):

- The low `b` bits hold the **main lesion ID**: 0 for none, 1 for the first
  main lesion, 2 for the second, and so on.
- The remaining `8 - b` bits are **overlay flags**, one per overlay, from the
  highest bit down: the first overlay is bit 7, the second bit 6, ...

So a file holds up to 2^b − 1 main lesions and 8 − b overlays: 63 and 2 with
the default split, 7 and 5 with `b = 3`.

| Pixel (binary, b = 6) | Meaning |
|---|---|
| `00 000000` | background |
| `00 000010` | second main lesion |
| `10 000000` | first overlay only |
| `10 000001` | first main lesion and first overlay |

## Metadata

The names and the split are stored as compressed text (a PNG `zTXt` chunk) with
the keyword `Lesions`, containing JSON:

```json
{"v": "1", "bc": 6, "mc": 2, "oc": 1, "mn": ["IRF", "SRF"], "on": ["HRF"]}
```

| Key | Meaning |
|---|---|
| `v` | Format version. |
| `bc` | Bits for main lesions (`b`). |
| `mc`, `oc` | Number of main lesions and overlays. |
| `mn`, `on` | Their names, in ID order and bit order. Optional. |

## Palette

The PNG has a palette, so image viewers show the lesions in color: each main
lesion has its own color, and set overlay bits blend a tint into it. The
palette is for display only; readers should use the pixel values.

## Reading a lesion file without octlabels

```python
import json

import numpy as np
from PIL import Image

with Image.open("patient_01/lesions/000.png") as image:
    pixels = np.array(image)
    meta = json.loads(image.text["Lesions"])

bits = meta["bc"]
main_ids = pixels & ((1 << bits) - 1)
masks = {}
for i, name in enumerate(meta.get("mn", [])):
    masks[name] = main_ids == i + 1
for k, name in enumerate(meta.get("on", [])):
    masks[name] = (pixels >> (7 - k)) & 1 == 1
print({name: int(mask.sum()) for name, mask in masks.items()})
```
