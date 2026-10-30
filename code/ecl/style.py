"""One visual system for every figure: hairline recessive axes, thin marks,
text in ink tokens, colour by role (never by rank).

Roles use the reference palette's first three categorical slots, which
validate all-pairs for colour-vision deficiency (dataviz skill, palette.md):
  MODEL  (blue)   the set of equally accurate models / model choice
  POLICY (orange) the reversion policies ASC 326 names
  ALT    (aqua)   a third series (the spline logit, or realized losses)
NOISE is a neutral grey for bootstrap refits, so it reads as background.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

MODEL, POLICY, ALT = "#2a78d6", "#eb6834", "#1baf7a"
MODEL_LIGHT = "#b7d3f6"
NOISE, NOISE_LIGHT = "#8c8a84", "#dcdad4"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e8e7e3"
FAMILY = {"lgbm": MODEL, "lgbm_mono": POLICY, "logit_spline": ALT}
FAMILY_LABEL = {"lgbm": "LightGBM", "lgbm_mono": "LightGBM, monotone", "logit_spline": "Spline logit"}


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.linewidth": 0.6, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 4, "ytick.major.pad": 4, "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round", "legend.frameon": False, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
        "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlepad": 6,
    })


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)
