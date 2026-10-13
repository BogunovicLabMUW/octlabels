# Lesions

Lesion masks are 2D (one value per pixel) and stored as one
[PNG per B-scan](../formats/lesions.md).

## Main lesions and overlays

Each `Lesion` is either a **main lesion** (mutually exclusive with the other
main lesions, e.g. fluid compartments) or an **overlay** (may overlap anything,
e.g. hyperreflective foci). The flag is stored on the lesion and kept when
loading and saving:

```python
import numpy as np
from octlabels import Lesion, LesionSet

bscan = LesionSet.load("patient_01/lesions/000.png")
print(bscan.main_lesion_names)            # ['IRF', 'SRF']
print(bscan["HRF"].main)                  # False: an overlay

height, width = bscan.height, bscan.width
ped = np.zeros((height, width), dtype=np.uint8)
ped[100:110, 30:40] = 1
bscan.add_lesion(Lesion("PED", ped, main=True))
bscan.save("out/000.png")
```

`main_lesion_names` can also be assigned; every other lesion becomes an overlay:

```python
bscan.main_lesion_names = ["IRF", "SRF", "PED"]
```

Saving checks that main lesions do not overlap.

## The bit split

With the default split of 6 bits, a file holds up to 63 main lesions and 2
overlays. More overlays need fewer bits for main lesions:

```python
bscan.save("out/000.png", main_lesion_bit_count=3)   # up to 7 main lesions, 5 overlays
reloaded = LesionSet.load("out/000.png")            # the split is read from the file
```

## Masks as arrays

```python
masks = bscan.to_array(["IRF", "SRF", "HRF", "PED"])   # [4, height, width], uint8
from_model = LesionSet.from_array(masks, ["IRF", "SRF", "HRF", "PED"],
                                  main_lesion_names=["IRF", "SRF", "PED"])
```

## The packing functions

The PNG format can also be used directly with arrays or dictionaries, with main
lesions and overlays passed separately. This is the lowest level and gives full
control over the file:

```python
from octlabels.lesions import packing

main = np.zeros((2, 128, 64), dtype=np.uint8)
main[0, 40:50] = 1
main[1, 60:70] = 1
overlays = np.zeros((1, 128, 64), dtype=np.uint8)
overlays[0, :, 10] = 1

packing.save_lesions("out/packed.png", main, overlays, ["IRF", "SRF"], ["HRF"])
main, overlays, main_names, overlay_names = packing.load_lesions("out/packed.png")

packing.save_lesion_dicts("out/packed.png", {"IRF": main[0], "SRF": main[1]},
                          {"HRF": overlays[0]})
main_dict, overlay_dict = packing.load_lesion_dicts("out/packed.png")
```

Loading decodes with the bit split stored in the file. Passing
`main_lesion_bit_count` overrides it; that still decodes correctly as long as
all IDs and overlays fit, and warns.
