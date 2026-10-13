# Custom properties

Besides `values`, every layer and region can carry any number of **properties**:
arrays with one entry per A-scan, such as `confidence`, `uncertainty` or your
own (a signal-to-noise ratio, the grader of each A-scan, whether an A-scan was
corrected by hand, ...). Their length is checked on every change, so they
always stay aligned with the A-scans.

## Adding properties

```python
import numpy as np
from octlabels import Layer, LayerSet

bscan = LayerSet.load("patient_01/layers/000.yml")
width = bscan.width
ilm = bscan["ILM"]

ilm.set_property("snr", np.linspace(0, 3, width).round(2))  # float
ilm.set_property("edited", np.zeros(width, dtype=np.uint8))  # 0/1 flag

# When creating annotations:
rpe = Layer("RPE2", np.full(width, 80.0), snr=np.ones(width))
bscan.add_layer(rpe)
bscan.add_layer_raw("BM2", np.full(width, 90.0), grader=np.full(width, 2))
```

For stacked model output, pass properties to `from_array` with the same shape
as the values ([layers, width] for a B-scan, [B-scans, layers, width] for a
volume):

```python
from octlabels import LayerVolume

heights = np.random.uniform(20, 100, size=(2, 3, 64))
volume = LayerVolume.from_array(
    heights, names=["ILM", "RPE", "BM"], info={"height": 128}, snr=np.ones_like(heights)
)
```

Rules:

- A property has exactly one entry per A-scan; anything else raises
  `ValueError`.
- Names are Python identifiers (letters, digits, `_`); `values` is reserved.
- `confidence` (uint8) and `uncertainty` (uint16, 0–999) have fixed types.
  Custom properties keep the type you give them.

## In the file

Each property becomes a row, aligned with the others:

```python
print(bscan.to_string()[:400])
```

```yaml
  - ILM:
    - values:   30,30.1,30.2,30.3,30.4,30.5,30.6,30.7, ...
    - confid:    1,   1,   1,   1,   1,   1,   1,   1, ...
    - snr:       0,0.05, 0.1,0.14,0.19,0.24,0.29,0.33, ...
    - edited:    0,   0,   0,   0,   0,   0,   0,   0, ...
```

Custom float properties are written exactly, so a float64 such as
`0.047619047619047616` takes 20 characters, and every row of the annotation is
padded to that width. Round values you do not need at full precision (as with
`.round(2)` above), or store them as float32.

Files store numbers, not types. When reading, a row of whole numbers becomes
`int64` and any other row `float64` (except `confidence` and `uncertainty`).

## Finding the properties of a file

```python
saved = LayerSet.from_string(bscan.to_string())
print(list(saved["ILM"].properties))          # ['confidence', 'snr', 'edited']
print(saved.layers.property_dtypes())         # every property of any layer, with its type

custom = {
    name: [key for key in layer.properties if key not in layer.PROPERTY_DTYPES]
    for name, layer in saved.layers.items()
}
print(custom)   # {'ILM': ['snr', 'edited'], 'RPE': [], ...}
```

## Using properties

```python
snr = saved["ILM"].properties["snr"]                    # [width]
stacked = saved.layers.stack("snr", names=["ILM"])      # [1, width]
saved["ILM"].remove_property("edited")

# ensure() gives missing layers every property, zero-filled, so all layers
# can be stacked:
complete = saved.ensure(["ILM", "OPL"])
print(complete.layers.stack("snr").shape)               # (2, width)
```

`layers.stack(key)` raises `KeyError` if a layer does not have the property;
use `ensure` or select layers with `names`.

## Properties in transforms

All transforms move properties together with the values. When A-scans are
resampled, float properties are interpolated linearly, integer properties take
the nearest value, and `confidence` takes the minimum of both neighbours (an
A-scan is only valid if its sources are valid).

To change how a property is resampled, subclass `Layer` and extend
`INTERPOLATION` (the modes are `"linear"`, `"nearest"` and `"min"`):

```python
from octlabels.layers import transforms


class SmoothUncertaintyLayer(Layer):
    INTERPOLATION = {**Layer.INTERPOLATION, "uncertainty": "linear"}


smooth = LayerSet({"height": 128})
smooth.add_layer(SmoothUncertaintyLayer("ILM", [30, 30], uncertainty=[0, 100]))
print(transforms.resize_horizontal(smooth, 4)["ILM"].uncertainty)   # [  0  25  75 100]
```

Files do not record the class, so annotations loaded from a file are plain
`Layer`s; apply the subclass where you create the annotations.
