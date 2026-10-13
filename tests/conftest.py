import numpy as np
import pytest

from octlabels import LayerSet


@pytest.fixture
def layer_set() -> LayerSet:
    """Two layers with uncertainty and a custom property, plus one region."""
    ls = LayerSet({"width": 8, "height": 20, "idx": 0})
    x = np.arange(8)
    ls.add_layer_raw("ILM", 5 + 0.5 * x, np.ones(8), uncertainty=x * 10, snr=x / 4)
    ls.add_layer_raw(
        "BM", 15 - 0.25 * x, [1, 1, 1, 0, 1, 1, 1, 1], uncertainty=x, snr=x / 8
    )
    ls.add_region_raw("DRUSEN", [0, 0, 1, 1, 1, 0, 0, 0], confidence=[200] * 8)
    return ls
