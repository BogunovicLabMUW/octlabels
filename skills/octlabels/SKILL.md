---
name: octlabels
description: Read, write and process OCT annotation files with the octlabels Python library - retinal layer boundaries and per-A-scan regions in NNN.yml files, lesion masks in NNN.png files, one file per B-scan in folders or zip files. Use when code works with OCT layer segmentation or lesion labels, octlabels, LayerSet, LayerVolume, LesionSet or LesionVolume, or converts such labels to or from NumPy arrays for training or evaluation.
---

# Using octlabels

octlabels stores OCT labels one file per B-scan:

- **Layer files** `NNN.yml` (or `.yml.gz`/`.yml.xz`): YAML with one row per
  annotation and property, one entry per A-scan. Layer boundaries (`Layer`,
  height per A-scan), regions (`AScanRegion`, 0/1 presence of a biomarker per
  A-scan) and properties such as `confidence` (uint8, 0 = invalid),
  `uncertainty` (uint16, 0–999) or custom ones.
- **Lesion files** `NNN.png`: 2D masks bit-packed into an 8-bit PNG. Main
  lesions (mutually exclusive, e.g. IRF, SRF, PED) and overlays (may overlap,
  e.g. HRF).
- **Volumes**: a folder of these files, or a `.zip` file of them. `NNN` is
  the B-scan index. Layers and lesions of a scan are separate volumes, usually
  `scan/layers/` and `scan/lesions/`.

## Choosing the API

- Need NumPy arrays (training, evaluation, plotting)? Use the array
  functions; they accept a single B-scan file or a whole volume.
- Need to inspect, edit, add properties, transform or validate? Load objects.
- Writing model output? Use `save_*_arrays` or `*.from_array(...).save(...)`.

## Arrays

```python
import octlabels

layer_names, values, confidence = octlabels.load_layer_arrays(
    "patient_01/layers/", ["ILM", "RPE", "BM"]
)  # values: float32 [B-scans, 3, width]; confidence: uint8, same shape
lesion_names, masks = octlabels.load_lesion_arrays(
    "patient_01/lesions/", ["IRF", "SRF", "HRF"]
)  # masks: uint8 0/1 [B-scans, 3, height, width]

octlabels.save_layer_arrays(
    "prediction/layers/", values, confidence, layer_names, info={"height": 128}
)
octlabels.save_lesion_arrays(
    "prediction/lesions/", masks, lesion_names, main_lesion_names=["IRF", "SRF"]
)
```

- A single B-scan file gives [layers, width] / [lesions, height, width].
- `names` selects and orders annotations; missing ones are zero. Always pass
  `names` for training data, so every sample has the same channels.
- Without `names`, all B-scans of a volume must have the same annotations,
  otherwise `ValueError`.
- Layer values have NaN replaced by 0; use `confidence > 0` as the valid mask.

## Objects

```python
import numpy as np
import octlabels
from octlabels import LayerVolume, LesionVolume, uncertainty_from_std

layers = octlabels.load("patient_01/layers/")        # LayerVolume
lesions = LesionVolume.load("patient_01/lesions/")
one = octlabels.load_bscan("patient_01/layers/", 2)  # B-scan index 2 only

bscan = layers[0]                       # by position
ilm = bscan["ILM"]                      # Layer: .values, .confidence, .properties
print(list(bscan.layers), list(bscan.regions), bscan.info)
srf = lesions[0]["SRF"]                 # Lesion: .mask [H, W], .main

values, confidence = bscan.to_array(["ILM", "RPE", "BM"])
fixed = bscan.ensure(["ILM", "RPE", "BM"])     # copy with exactly these layers

ilm.set_property("snr", np.zeros(bscan.width))  # custom per-A-scan property
bscan.save("out/000.yml.xz")                   # compression from the file name
layers.save("out/layers.zip", compress="lzma") # whole volume as one zip

prediction = LayerVolume.from_array(
    np.random.uniform(20, 100, size=(2, 3, 64)),
    names=["ILM", "RPE", "BM"],
    info={"height": 128, "source": "model"},
    uncertainty=uncertainty_from_std(np.full((2, 3, 64), 2.0)),
)
```

Sets (`LayerSet`, `LesionSet`) and volumes (`LayerVolume`, `LesionVolume`)
share `load`, `save`, `to_array(names)`, `from_array`, `ensure`, `subset`,
`copy`; volumes also `load_bscan`. `bscan.layers`, `.regions` and `.lesions`
are dict-like (`in`, `del`, `.items()`, `.rename({...})`).

## Transforms

```python
from octlabels.layers import transforms

augmented = transforms.rotate(bscan, np.deg2rad(5))
resized = transforms.resize_horizontal(bscan, 32)
```

Also `flip`, `resize_vertical`, `spatial_pad_end`, `shear`, `scale`, `affine`.
They return copies, keep all arrays aligned, set confidence 0 where a boundary
leaves the image, and update `lateral_resolution_um`/`axial_resolution_um` in
`info`. Lesion masks are not transformed by these functions.

## Pitfalls

- `octlabels.load_bscan(path, idx)` and `LayerVolume.load_bscan` take the
  B-scan **index** from the file name; `volume[i]` takes the **position**.
- Every property must have one entry per A-scan; region values must be 0 or 1;
  main lesions must not overlap. Violations raise `ValueError`.
- Custom float properties are written exactly; round them (or use float32) to
  keep files small.
- `info` accepts any key; `LayerSet.validate()` checks the documented ones
  (`width`, `height`, `idx`, `axial_resolution_um`, `lateral_resolution_um`,
  `bscan_spacing_um`, `laterality` "OD"/"OS", `subject_id`, `scan_id`,
  `device`, `source` "manual"/"model"/"corrected", `grader`, `model`,
  `format_version`; see `octlabels.INFO_KEYS`). Never put direct patient
  identifiers into `info`.
- Lesion PNGs store only masks and names, not `info`.
- For separate main/overlay arrays use `octlabels.lesions.packing`
  (`save_lesions`, `load_lesions`, `save_lesion_dicts`, `load_lesion_dicts`).
- Reading `.mat` files (`octlabels.layers.matlab.load_duke_mat`) needs
  `pip install "octlabels[matlab]"`.

## Reading the files without octlabels

Layer files are plain YAML; each row is comma-separated text (or a YAML list):

```python
import yaml

with open("patient_01/layers/000.yml") as file:
    data = yaml.safe_load(file)
first_layer = data["layers"][0]          # {"ILM": [{"values": "30,30.1,..."}, ...]}
```

Lesion PNGs: the low bits of each pixel are the main lesion ID (1 = first),
overlay `k` is bit `7 - k`; names and the bit split are JSON in the PNG text
chunk `"Lesions"` (`{"bc": 6, "mn": [...], "on": [...]}`).
