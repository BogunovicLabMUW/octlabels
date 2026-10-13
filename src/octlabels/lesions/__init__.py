"""Lesion masks of OCT B-scans, packed into one PNG per B-scan.

The array-level functions of the PNG format (``save_lesions``,
``load_lesions``, ``save_lesion_dicts``, ``load_lesion_dicts``) are in
[`octlabels.lesions.packing`][octlabels.lesions.packing].
"""

from octlabels.lesions import packing
from octlabels.lesions.lesion_set import Lesion, LesionSet
from octlabels.lesions.lesion_volume import LesionVolume

__all__ = ["Lesion", "LesionSet", "LesionVolume", "packing"]
