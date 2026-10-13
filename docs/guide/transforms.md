# Transforms and segmentation maps

## Transforms

The functions in `octlabels.layers.transforms` return a new `LayerSet`. Layers,
regions and every property are transformed together, so they stay aligned.

```python
import numpy as np
from octlabels import LayerSet
from octlabels.layers import transforms

bscan = LayerSet.load("patient_01/layers/000.yml")

flipped = transforms.flip(bscan)
narrow = transforms.resize_horizontal(bscan, 32)
taller = transforms.resize_vertical(bscan, 256)
padded = transforms.spatial_pad_end(bscan, (160, 80))   # (height, width)
rotated = transforms.rotate(bscan, np.deg2rad(5))
sheared = transforms.shear(bscan, 0.05, 0.0)
scaled = transforms.scale(bscan, 1.1, 1.0)
augmented = transforms.affine(
    bscan, angle_radians=0.05, shear=(0.05, 0), translation=(3, -2), scale=(1.1, 1)
)
cropped = bscan[:, 10:50]   # A-scans 10 to 49
```

Notes on `affine` and its shortcuts:

- Points are transformed around the image center unless `center` is given.
- Invalid A-scans (confidence 0) are bridged by interpolation before
  transforming; afterwards, confidence is 0 where a boundary leaves the image.
- A steep boundary can fold back under rotation or shear, so that a column is
  crossed more than once. Such columns have no single height and get
  confidence 0.
- Transforms that reverse the A-scan order (e.g. rotations of 90° or more)
  raise `ValueError`.
- See [Metadata](metadata.md#how-transforms-update-the-metadata) for how the
  resolutions in `info` are updated.

## Segmentation maps

Some models predict per-pixel classes instead of boundary heights.
`octlabels.layers.rasterize` converts between the two:

```python
from octlabels.layers import rasterize

bands, band_confidence = rasterize.generate_2d_segmentation(bscan)
print(bands.shape)            # [layers + 1, height, width]: above, between, below

lines = rasterize.generate_boundary_maps(bscan)
print(lines.shape)            # [layers, height, width]: one-pixel boundary lines

positions, confidence = rasterize.extract_boundaries_from_maps(lines)
print(positions.shape)        # [layers, width]
```

Band `i` covers the pixels between layer `i - 1` and layer `i`; a band column
is only filled where both bounding layers are valid.
