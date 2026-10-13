# Regions

A **region** (`AScanRegion`) records per A-scan whether a biomarker is present:
`values` is 1 where it is present and 0 elsewhere. Regions are stored next to
the layers of a B-scan, in the `regions` section of the
[layer file](../formats/layers.md).

```python
import numpy as np
from octlabels import AScanRegion, LayerSet

bscan = LayerSet.load("patient_01/layers/000.yml")
width = bscan.width

presence = np.zeros(width, dtype=np.uint8)
presence[20:30] = 1
bscan.add_region_raw("IRF", presence)
bscan.add_region(AScanRegion("PED", np.zeros(width), confidence=np.full(width, 200)))

print(list(bscan.regions))            # ['DRUSEN', 'IRF', 'PED']
print(bscan.regions["IRF"].values[18:32])
```

Compared with layers:

- `values` must be 0 or 1 (stored as uint8); anything else raises
  `ValueError`, also when assigned later.
- `confidence` is optional.
- Regions are a separate namespace: `bscan["ILM"]` always returns a layer, a
  region of the same name is `bscan.regions["ILM"]`.

Regions support everything layers do: properties, `ensure` and `subset`
(with `region_names`), stacking and transforms. Transforms resample region
values with the nearest A-scan, so they stay 0 or 1.

```python
stacked = bscan.regions.stack()                     # [regions, width], uint8
selected = bscan.ensure(["ILM"], region_names=["IRF", "SRF"])
print(list(selected.regions))                       # ['IRF', 'SRF']; SRF is empty
```
