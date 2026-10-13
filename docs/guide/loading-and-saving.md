# Loading and saving

## The classes

Layers and lesions are organised the same way:

| | Layers | Lesions |
|---|---|---|
| one annotation | `Layer` (heights), `AScanRegion` (presence per A-scan) | `Lesion` (2D mask) |
| one B-scan | `LayerSet`: `info`, `layers`, `regions` | `LesionSet`: `info`, `lesions` |
| volume | `LayerVolume` | `LesionVolume` |

The B-scan and volume classes share their main methods: `load`, `save`,
`to_array`, `from_array`, `ensure`, `subset` and `copy`; volumes also have
`load_bscan`.

## Loading

`octlabels.load` picks the class from the path: a `.yml[.gz|.xz]` file gives a
`LayerSet`, a `.png` file a `LesionSet`, and a folder or `.zip` file the
matching volume.

```python
import octlabels
from octlabels import LayerSet, LayerVolume, LesionSet, LesionVolume

layers = octlabels.load("patient_01/layers/")     # LayerVolume
lesions = LesionVolume.load("patient_01/lesions/")
bscan = LayerSet.load("patient_01/layers/000.yml")
masks = LesionSet.load("patient_01/lesions/000.png")
```

To read one B-scan of a volume without reading the others, use `load_bscan`.
It takes the B-scan index (the number in the file name), not the position:

```python
bscan = octlabels.load_bscan("patient_01/layers/", 2)
assert bscan.info["idx"] == 2
```

## Accessing annotations

A B-scan behaves like a small dictionary of annotations:

```python
bscan = layers[0]                  # volumes are indexed by position
ilm = bscan["ILM"]                 # a Layer; also bscan.layers["ILM"]
print(ilm.values[:5])              # heights, float32
print(ilm.confidence[:5])          # uint8, 0 = invalid
print(list(bscan.layers), list(bscan.regions))

srf = lesions[0]["SRF"]            # a Lesion
print(srf.mask.shape, srf.main)    # [height, width] uint8, main lesion?

for bscan in layers:               # volumes are iterable
    print(bscan.info["idx"], bscan.width)
```

`bscan.layers`, `bscan.regions` and `bscan.lesions` are dict-like: use `in`,
`del`, `.items()` and `.rename({"old": "new"})`.

## Saving

```python
layers.save("out/layers/")                       # folder of .yml files
layers.save("out/layers/", compress="lzma")      # folder of .yml.xz files
layers.save("out/layers.zip")                    # one zip file (Deflate)
layers.save("out/layers.zip", compress="lzma")   # one zip file, smallest
lesions.save("out/lesions/")

layers[0].save("out/single/000.yml.xz")          # compression from the name
layers[0].save("out/single/000.yml", yaml_arrays=True)
```

`yaml_arrays=True` writes rows as YAML lists instead of comma-separated text,
see [Layer files](../formats/layers.md#row-styles). Volumes name their files
after `info["idx"]` and refuse to save two B-scans with the same index.
