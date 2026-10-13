# Metadata (`info`)

Every B-scan has an `info` dictionary. It accepts any keys and is stored at the
top of [layer files](../formats/layers.md); lesion files store only the masks
and their names.

```python
from octlabels import LayerSet

bscan = LayerSet.load("patient_01/layers/000.yml")
print(bscan.info)    # {'width': 64, 'height': 128, 'idx': 0, ...}
```

`width` is set automatically from the annotations; `idx` is set when volumes
are created or loaded.

## Built-in keys

The following keys have a defined meaning. All are optional. They are checked
by `LayerSet.validate()`, and transforms keep the resolutions correct.

| Key | Type | Meaning |
|---|---|---|
| `width` | integer > 0 | Number of A-scans. |
| `height` | integer > 0 | Pixels per A-scan. |
| `idx` | integer ≥ 0 | Index of the B-scan in its volume. |
| `axial_resolution_um` | number > 0 | Micrometres per pixel along the A-scan. |
| `lateral_resolution_um` | number > 0 | Micrometres between neighbouring A-scans. |
| `bscan_spacing_um` | number > 0 | Micrometres between neighbouring B-scans. |
| `laterality` | `"OD"` or `"OS"` | Right or left eye. |
| `subject_id` | text | Pseudonymous subject identifier. |
| `scan_id` | text | Identifier of the scan. |
| `device` | text | Acquisition device, e.g. `"Spectralis"`. |
| `source` | `"manual"`, `"model"` or `"corrected"` | Origin of the annotations. |
| `grader` | text | Who annotated the B-scan. |
| `model` | text | Model name and version that produced the annotations. |
| `format_version` | integer > 0 | Version of the file format. |

The same list is available in code, with a description and a check for each
key:

```python
from octlabels import INFO_KEYS

print(INFO_KEYS["laterality"].description)       # Eye: 'OD' (right) or 'OS' (left).
print(INFO_KEYS["laterality"].is_valid("left"))  # False
```

!!! warning "No patient identifiers"
    Files are meant to be shared. Do not store names, birth dates, hospital IDs
    or other direct identifiers; use pseudonymous `subject_id`s.

## Setting and validating

```python
bscan.info.update(
    axial_resolution_um=3.87,
    lateral_resolution_um=11.3,
    laterality="OD",
    source="manual",
    grader="BF",
)
bscan.validate()               # checks the built-in keys and value ranges

bscan.info["laterality"] = "left"
try:
    bscan.validate()
except ValueError as error:
    print(error)   # info['laterality'] = 'left' is invalid. Eye: 'OD' (right) or 'OS' (left).
bscan.info["laterality"] = "OD"
```

`validate()` also requires `width` and `height` and checks that every layer
lies within `[0, height]`.

## Your own keys

Any other key is kept as it is and written to the file. Values may be numbers,
text, booleans or `None`; strings that YAML would misread (such as `"yes"` or
`"1"`) are quoted automatically.

```python
bscan.info["study"] = "AMD-2026"
bscan.info["reviewed"] = True
assert LayerSet.from_string(bscan.to_string()).info["reviewed"] is True
```

## How transforms update the metadata

Transforms that change the pixel size update the resolutions:

```python
from octlabels.layers import transforms

half = transforms.resize_horizontal(bscan, bscan.width // 2)
print(half.info["lateral_resolution_um"])     # 22.6: each A-scan covers twice as much
taller = transforms.resize_vertical(bscan, bscan.height * 2)
print(taller.info["axial_resolution_um"])     # 1.935
zoomed = transforms.scale(bscan, 2.0, 1.0)
print(zoomed.info["lateral_resolution_um"])   # 5.65
```

Rotation, shear, translation, flipping, padding and cropping leave the
resolutions unchanged. `flip` also leaves `laterality` unchanged: whether a
mirrored B-scan resembles the other eye depends on the scan direction, which
the file does not record.

## Volumes

`volume.info` returns a copy of the first B-scan's info. To set a key for every
B-scan, loop over the volume:

```python
from octlabels import LayerVolume

volume = LayerVolume.load("patient_01/layers/")
for bscan in volume:
    bscan.info["subject_id"] = "S001"
print(volume.info["subject_id"])
```
