"""octlabels: annotations of OCT scans (retinal layers, biomarker regions,
lesion masks)."""

from octlabels import layers, lesions
from octlabels.info import INFO_KEYS, InfoKey
from octlabels.layers import (
    AnnotationGroup,
    AScanAnnotation,
    AScanRegion,
    Layer,
    LayerSet,
    LayerVolume,
    uncertainty_from_std,
)
from octlabels.lesions import Lesion, LesionSet, LesionVolume
from octlabels.loading import (
    LayerArrays,
    LesionArrays,
    load,
    load_bscan,
    load_layer_arrays,
    load_lesion_arrays,
    save_layer_arrays,
    save_lesion_arrays,
)

__version__ = "0.1.0"

__all__ = [
    "INFO_KEYS",
    "AScanAnnotation",
    "AScanRegion",
    "AnnotationGroup",
    "InfoKey",
    "Layer",
    "LayerArrays",
    "LayerSet",
    "LayerVolume",
    "Lesion",
    "LesionArrays",
    "LesionSet",
    "LesionVolume",
    "layers",
    "lesions",
    "load",
    "load_bscan",
    "load_layer_arrays",
    "load_lesion_arrays",
    "save_layer_arrays",
    "save_lesion_arrays",
    "uncertainty_from_std",
]
