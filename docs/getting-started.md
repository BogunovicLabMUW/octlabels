# Getting started

## Installation

```bash
pip install git+https://github.com/BogunovicLabMUW/octlabels.git
```

The library needs `numpy`, `pyyaml` and `pillow`. Reading MATLAB files needs
the optional extra `pip install "octlabels[matlab]"`.

## Folder layout

Annotations are stored one file per B-scan, named after the B-scan index. A
typical scan looks like this:

```text
patient_01/
  layers/            000.yml, 001.yml, ...  (or .yml.gz / .yml.xz)
  lesions/           000.png, 001.png, ...
```

Instead of a folder, a volume can also be one zip file of the same files
(`layers.zip`).

## Arrays in one call

Most code needs the annotations as NumPy arrays. One function per kind loads a
single B-scan file or a whole volume:

```python
import octlabels

names, values, confidence = octlabels.load_layer_arrays("patient_01/layers/000.yml")
names, values, confidence = octlabels.load_layer_arrays("patient_01/layers/")
names, masks = octlabels.load_lesion_arrays("patient_01/lesions/000.png")
names, masks = octlabels.load_lesion_arrays("patient_01/lesions/", ["IRF", "SRF", "HRF"])
```

| Array | Layers | Lesions |
|---|---|---|
| one B-scan | `values`, `confidence`: [layers, width] | `masks`: [lesions, height, width] |
| volume | [B-scans, layers, width] | [B-scans, lesions, height, width] |

Passing `names` selects and orders the annotations; missing ones are zero
(layers: value 0, confidence 0; lesions: empty masks). Without `names`, all
B-scans of a volume must contain the same annotations.

Arrays are saved the same way; the path decides between a single B-scan file
and a volume:

```python
octlabels.save_layer_arrays("prediction/layers/", values, confidence, names)
octlabels.save_lesion_arrays(
    "prediction/lesions/", masks, ["IRF", "SRF", "HRF"], main_lesion_names=["IRF", "SRF"]
)
```

## Objects

For anything beyond arrays, load the annotations as objects:

```python
volume = octlabels.load("patient_01/layers/")        # a LayerVolume
bscan = octlabels.load_bscan("patient_01/layers/", 1)  # one LayerSet, reads one file
ilm = bscan["ILM"]
print(ilm.values[:5], ilm.confidence[:5])
```

Continue with the [user guide](guide/loading-and-saving.md), or read about the
[file formats](formats/layers.md).
