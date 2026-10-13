"""Draws the octlabels logo: retinal layers with the foveal pit and a pigment
epithelial detachment (PED), the RPE lifted off Bruch's membrane.

Writes docs/assets/logo.svg. The favicon (docs/assets/favicon.png) is this
logo rendered at 64 x 64 pixels.
"""

import math
from pathlib import Path

BLUE, ORANGE, SURFACE = "#2a78d6", "#eb6834", "#ffffff"
SIZE = 128
LEFT, RIGHT = 14, SIZE - 14
FOVEA_X = 46
PED_START, PED_END = 58, 102  # x range of the detachment


def ped_lift(x: float) -> float:
    """0 outside the detachment, rising smoothly to 1 at its center."""
    if not PED_START < x < PED_END:
        return 0.0
    phase = (x - PED_START) / (PED_END - PED_START)
    return 1 - abs(2 * phase - 1) ** 2.5  # round top, steep sides


def smooth_lift(x: float, radius: int) -> float:
    """``ped_lift`` averaged over ``x - radius .. x + radius``: rounds its corners."""
    offsets = range(-radius, radius + 1)
    return sum(ped_lift(x + offset) for offset in offsets) / len(offsets)


def fovea_dip(x: float, depth: float, width: float) -> float:
    return depth * math.exp(-(((x - FOVEA_X) / width) ** 2))


def line(height_at) -> str:
    points = []
    for step in range(101):
        x = LEFT + step * (RIGHT - LEFT) / 100
        points.append(f"{x:.1f},{height_at(x):.1f}")
    return "M" + " L".join(points)


def ilm(x: float) -> float:
    # The inner retina follows the detachment more gently than the RPE.
    return 36 + fovea_dip(x, depth=20, width=16) - 5 * smooth_lift(x, radius=14)


def rpe(x: float) -> float:
    return 86 - 28 * smooth_lift(x, radius=6)


BRUCHS_MEMBRANE_Y = 96

# The PED: the space between the lifted RPE and Bruch's membrane.
dome = [
    f"{x:.1f},{rpe(x):.1f}"
    for x in (PED_START + step * (PED_END - PED_START) / 60 for step in range(61))
]
ped = (
    f"M{PED_START},{BRUCHS_MEMBRANE_Y} L"
    + " L".join(dome)
    + f" L{PED_END},{BRUCHS_MEMBRANE_Y} Z"
)

white_line = (
    f'fill="none" stroke="{SURFACE}" stroke-width="7" '
    'stroke-linecap="round" stroke-linejoin="round"'
)
bruchs_membrane = f"M{LEFT},{BRUCHS_MEMBRANE_Y} L{RIGHT},{BRUCHS_MEMBRANE_Y}"
svg = "\n".join(
    [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}"'
        ' role="img" aria-label="octlabels logo">',
        f'  <rect width="{SIZE}" height="{SIZE}" rx="26" fill="{BLUE}"/>',
        f'  <path d="{ped}" fill="{ORANGE}"/>',
        f'  <path d="{line(ilm)}" {white_line}/>',
        f'  <path d="{line(rpe)}" {white_line}/>',
        f'  <path d="{bruchs_membrane}" {white_line}/>',
        "</svg>",
        "",
    ]
)

out = Path(__file__).parent.parent / "docs" / "assets" / "logo.svg"
out.write_text(svg)
print("written", out)
