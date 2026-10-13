"""Runs the Python examples of the documentation, llms.txt and the Claude skill.

All ```python blocks of a page run in order, in one namespace, in a folder that
contains a small sample scan ``patient_01/``. A block preceded by a line
``<!-- skip-test: reason -->`` is skipped.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from octlabels import INFO_KEYS, LayerVolume, LesionVolume

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
PAGES = [
    *sorted(DOCS.rglob("*.md")),
    DOCS / "llms.txt",
    *sorted((ROOT / "skills").rglob("SKILL.md")),
]
BLOCK = re.compile(r"(<!-- skip-test[^>]*-->\s*)?```python\n(.*?)```", re.DOTALL)

HEIGHT, WIDTH, NUM_BSCANS = 128, 64, 3


def _write_sample_scan(folder: Path) -> None:
    """Three B-scans with layers ILM, RPE, BM, a DRUSEN region, and lesions
    IRF and SRF (main) and HRF (overlay)."""
    columns = np.arange(WIDTH)
    heights = np.stack([30 + 0.1 * columns, 70 + 0.05 * columns, 80 + 0.05 * columns])
    values = np.repeat(heights[np.newaxis], NUM_BSCANS, axis=0)
    layers = LayerVolume.from_array(
        values, names=["ILM", "RPE", "BM"], info={"height": HEIGHT}
    )
    for layer_set in layers:
        drusen = np.zeros(WIDTH, dtype=np.uint8)
        drusen[25:35] = 1
        layer_set.add_region_raw("DRUSEN", drusen)
    layers.save(folder / "layers")

    masks = np.zeros((NUM_BSCANS, 3, HEIGHT, WIDTH), dtype=np.uint8)
    masks[:, 0, 40:50, 10:30] = 1  # IRF
    masks[:, 1, 60:70, 20:40] = 1  # SRF
    masks[:, 2, :, 12] = 1  # HRF
    lesions = LesionVolume.from_array(
        masks, ["IRF", "SRF", "HRF"], main_lesion_names=["IRF", "SRF"]
    )
    lesions.save(folder / "lesions")


def _examples(page: Path) -> list[str]:
    return [code for skip, code in BLOCK.findall(page.read_text()) if not skip]


@pytest.mark.parametrize("page", PAGES, ids=lambda page: str(page.relative_to(ROOT)))
def test_examples_run(page, tmp_path, monkeypatch):
    examples = _examples(page)
    if not examples:
        pytest.skip("no Python examples")
    _write_sample_scan(tmp_path / "patient_01")
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    for number, code in enumerate(examples, start=1):
        try:
            exec(compile(code, f"{page.name} example {number}", "exec"), namespace)
        except Exception as error:
            pytest.fail(f"{page.name}, example {number} failed: {error!r}\n\n{code}")


def test_metadata_page_lists_every_built_in_key():
    page = (DOCS / "guide" / "metadata.md").read_text()
    for key in INFO_KEYS:
        assert f"`{key}`" in page, key
