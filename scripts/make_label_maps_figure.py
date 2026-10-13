"""Draws docs/assets/label-maps.svg: in a combined label map a lesion hides the
boundaries underneath; octlabels keeps boundaries, validity and lesions apart."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Reference palette (dataviz skill): first three categorical slots, valid all-pairs.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"

WIDTH, HEIGHT = 200, 100
x = np.arange(WIDTH)
fovea = 14 * np.exp(-(((x - 70) / 22) ** 2))
ilm = 22 + fovea * 0.9
rpe = 62 - 0.01 * (x - 100) ** 2 / 40
bm = rpe + 7

# A fluid lesion between the photoreceptors and the RPE, lifting nothing (schematic).
lesion = np.zeros((HEIGHT, WIDTH), dtype=bool)
rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
# Center between pixel rows/columns, so the outline has no single-pixel tips.
lesion_field = ((cols - 140.5) / 26) ** 2 + ((rows - 63.5) / 9.3) ** 2
lesion |= lesion_field <= 1

# Panel A: one class per pixel. 0 background, 1 retina, 2 RPE-BM band, 3 lesion.
label_map = np.zeros((HEIGHT, WIDTH), dtype=int)
label_map[(rows >= ilm) & (rows < rpe)] = 1
label_map[(rows >= rpe) & (rows < bm)] = 2
label_map[lesion] = 3  # painted last: covers the RPE and BM boundaries

# Panel B: confidence per A-scan; one stretch marked invalid by the grader.
confidence = np.ones(WIDTH, dtype=int)
confidence[16:46] = 0

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, (left, right) = plt.subplots(
    1, 2, figsize=(10.5, 4.2), facecolor=SURFACE, gridspec_kw={"wspace": 0.12}
)

# --- Left: combined label map --------------------------------------------------
cmap = ListedColormap([SURFACE, BLUE, AQUA, ORANGE])
left.imshow(
    label_map, cmap=cmap, vmin=0, vmax=3, interpolation="nearest", aspect="auto"
)
left.set_title("Combined label map (PNG)", loc="left", color=INK, fontsize=12, pad=10)
left.annotate(
    "RPE and BM here: missing,\nor hidden by the lesion?",
    xy=(140, 64),
    xytext=(108, 90),
    color=INK,
    fontsize=9.5,
    ha="left",
    arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 1},
)
left.text(8, 40, "retina", color="white", fontsize=9.5, weight="bold")
left.legend(
    handles=[
        Patch(color=BLUE, label="retina"),
        Patch(color=AQUA, label="RPE–BM"),
        Patch(color=ORANGE, label="lesion (SRF)"),
    ],
    loc="upper right",
    frameon=False,
    fontsize=8.5,
)

# --- Right: octlabels ----------------------------------------------------------
right.set_facecolor(SURFACE)
lesion_rgba = np.zeros((HEIGHT, WIDTH, 4))
lesion_rgba[lesion] = matplotlib.colors.to_rgba(ORANGE, 0.35)
right.imshow(lesion_rgba, interpolation="nearest", aspect="auto")
right.contour(lesion_field, levels=[1.0], colors=[ORANGE], linewidths=1.5)
valid = confidence > 0
invalid_span = (x >= 15) & (x <= 46)  # include the neighbours so the dashes connect
for name, heights in (("ILM", ilm), ("RPE", rpe), ("BM", bm)):
    right.plot(x, np.where(valid, heights, np.nan), color=INK, lw=2)
    right.plot(
        x,
        np.where(invalid_span, heights, np.nan),
        color=INK_2,
        lw=2,
        ls=(0, (1.5, 1.5)),
    )
    right.text(WIDTH + 4, heights[-1], name, color=INK, fontsize=9, va="center")
right.annotate(
    "layers stay complete under the lesion",
    xy=(150, rpe[150]),
    xytext=(96, 90),
    color=INK,
    fontsize=9.5,
    arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 1},
)
right.annotate(
    "confidence 0:\nexplicitly invalid",
    xy=(30, ilm[30]),
    xytext=(4, 6),
    color=INK,
    fontsize=9.5,
    va="top",
    arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 1},
)
right.text(150, 46, "lesion in its own file", color=INK, fontsize=9, ha="center")
right.set_title(
    "octlabels: boundaries, validity and lesions kept apart",
    loc="left",
    color=INK,
    fontsize=12,
    pad=10,
)
right.set_xlim(0, WIDTH + 20)
right.set_ylim(HEIGHT, 0)

for axes in (left, right):
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color(MUTED)

out = Path(__file__).parent.parent / "docs" / "assets" / "label-maps.svg"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("written", out)
