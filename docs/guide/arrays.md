# Arrays and training

## From files to arrays

```python
import octlabels

names, values, confidence = octlabels.load_layer_arrays(
    "patient_01/layers/", ["ILM", "RPE", "BM"]
)
names, masks = octlabels.load_lesion_arrays("patient_01/lesions/", ["IRF", "SRF", "HRF"])
print(values.shape, masks.shape)   # [B-scans, 3, width], [B-scans, 3, height, width]
```

Layer values are float32 with NaN replaced by 0; confidence and masks are
uint8. The same works on the objects:

```python
from octlabels import LayerVolume

layers = LayerVolume.load("patient_01/layers/")
values, confidence = layers.to_array(["ILM", "RPE", "BM"])   # whole volume
values, confidence = layers[0].to_array(["ILM", "RPE", "BM"]) # one B-scan
```

## Fixed sets of annotations: `ensure`

Training needs the same channels for every sample, but real data is
inconsistent: some B-scans miss a layer, some files have extra ones. `ensure`
returns a copy with exactly the given annotations, in that order; missing ones
are zero (layers) or empty (lesions). `to_array(names)` does the same in one
step.

```python
bscan = layers[0].ensure(["ILM", "RPE", "BM", "OPL"])
print(list(bscan.layers))           # ['ILM', 'RPE', 'BM', 'OPL']
print(bscan["OPL"].confidence.max())  # 0: OPL was missing
```

Missing annotations get every property the others have (zero-filled), so
stacking any property works afterwards.

## From model output to files

```python
import numpy as np
from octlabels import LayerVolume, LesionVolume, uncertainty_from_std

predicted_heights = np.random.uniform(20, 100, size=(3, 3, 64))   # [B, layers, W]
predicted_std = np.random.uniform(0.5, 5, size=(3, 3, 64))
predicted_masks = np.zeros((3, 3, 128, 64), dtype=np.uint8)       # [B, lesions, H, W]
predicted_masks[:, 0, 40:50] = 1
predicted_masks[:, 2, :, 10] = 1

layers = LayerVolume.from_array(
    predicted_heights,
    confidence=np.ones_like(predicted_heights),
    names=["ILM", "RPE", "BM"],
    info={"height": 128, "source": "model", "model": "unet-v3"},
    uncertainty=uncertainty_from_std(predicted_std),
)
lesions = LesionVolume.from_array(
    predicted_masks, names=["IRF", "SRF", "HRF"], main_lesion_names=["IRF", "SRF"]
)
layers.save("prediction/layers/")
lesions.save("prediction/lesions/")
```

`uncertainty_from_std` converts standard deviations in pixels to the stored
uncertainty encoding (clamped to 1–100, mapped to 0–999).

## A PyTorch dataset

<!-- skip-test: needs torch -->
```python
import torch
from torch.utils.data import Dataset
from octlabels import LayerVolume, LesionVolume


class OCTDataset(Dataset):
    def __init__(self, scan: str, layer_names: list[str], lesion_names: list[str]):
        self.layers = LayerVolume.load(f"{scan}/layers/")
        self.lesions = LesionVolume.load(f"{scan}/lesions/")
        self.layer_names = layer_names
        self.lesion_names = lesion_names

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int):
        values, confidence = self.layers[idx].to_array(self.layer_names)
        masks = self.lesions[idx].to_array(self.lesion_names)
        return (
            torch.from_numpy(values),
            torch.from_numpy(confidence).float(),
            torch.from_numpy(masks).float(),
        )
```
