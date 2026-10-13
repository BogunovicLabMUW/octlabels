# MATLAB import

Several public OCT datasets, such as the Duke DME and AMD datasets, store layer
boundaries as one numeric array in a MATLAB `.mat` file.
`octlabels.layers.matlab` reads them into a `LayerVolume`.

This needs optional packages: `pip install "octlabels[matlab]"` (scipy for
MATLAB v5 files, h5py for v7.3 files). They are only imported when you call the
function, and an error tells you how to install them if they are missing.

<!-- skip-test: needs .mat files -->
```python
from octlabels.layers import matlab

volume = matlab.load_duke_mat("Subject_01.mat")              # Duke DME: manualLayers1
volume = matlab.load_duke_mat("Subject_01.mat", variable="manualLayers2")
volume = matlab.load_duke_mat("AMD_Subject_1001.mat", height=512)  # Duke AMD: layerMaps

# Any file with one boundary array; axes is its dimension order
# (L = boundaries, A = A-scans, B = B-scans):
volume = matlab.load_duke_mat(
    "my.mat", variable="boundaries", axes="BLA", names=["ILM", "RPE", "BM"]
)
volume.save("converted/")
```

What happens on import:

- NaN boundaries become value 0 with confidence 0.
- MATLAB's 1-based rows become 0-based (`one_based=False` turns this off).
- B-scans without any annotation are skipped (`skip_unannotated=False` keeps
  them); `info["idx"]` keeps the original index.
- The height is taken from an `images` array in the file, if there is one.
- If the variable is not found, the error lists the variables in the file.

The known variables and their layouts are in
[`DUKE_VARIABLES`][octlabels.layers.matlab.DUKE_VARIABLES].
