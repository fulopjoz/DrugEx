"""Publication figure style for the EGCG ligand-based RL pilot.

Local copy of ``enablement/figures/_figstyle.py`` (the canonical version lives on
the campaign branches at
``ccr2_gen/ablation/analysis/final_selection/enablement/figures/_figstyle.py``
and is NOT on this branch). Kept local per the self-contained layout ruling;
it can therefore drift from the canonical one.

Two deliberate deviations from the canonical copy:
  * No ``matplotlib.use("Agg")``. The canonical module forces a headless backend
    at import, which would suppress inline display in an interactive notebook.
    ``nbconvert --execute`` works fine with the default backend.
  * Adds an ARM palette (baseline vs reinforced), which the campaign copy has no
    need for.

Hard rules inherited unchanged:
  * ONE plot per figure (MODULAR) — no composite grids. Panels saved separately.
  * NO on-canvas title/suptitle, NO annotation boxes, NO claim-text on the canvas.
    The message lives in the CAPTION, never the plot.
  * Axis labels carry units + direction.
  * High resolution (savefig 350 dpi), PNG + PDF.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent / "panels"

# Arm palette: baseline = muted cool, reinforced = warm. Encodes before/after.
ARM = {"baseline": "#4363d8", "reinforced": "#f58231"}
ARM_LABEL = {"baseline": "Pretrained (baseline)", "reinforced": "Reinforced (final epoch)"}
ARM_ORDER = ["baseline", "reinforced"]

INK = "#2b2f36"
MUTED = "#8a9099"
ZERO = "#b0b6bd"


def apply_style():
    """Global rcParams — call once at the top of the notebook."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 350,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10.5,
        "axes.titlesize": 11, "axes.titleweight": "regular",
        "axes.labelsize": 11, "axes.labelcolor": INK,
        "axes.edgecolor": "#3a3f45", "axes.linewidth": 0.8,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.color": INK, "ytick.color": INK,
        "legend.fontsize": 9, "legend.frameon": False, "legend.handlelength": 1.2,
        "grid.color": "#e3e7ec", "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    })


def new_ax(w=3.6, h=3.1, yonly_grid=True):
    """One modular panel (journal single-column ~3.4-3.6 in). Returns (fig, ax)."""
    fig, ax = plt.subplots(figsize=(w, h))
    sns.despine(ax=ax)
    if yonly_grid:
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", visible=True)
    return fig, ax


def arm_legend(ax, loc="best", title=None):
    """Compact baseline/reinforced legend."""
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ARM[a], edgecolor="none", label=ARM_LABEL[a])
               for a in ARM_ORDER]
    ax.legend(handles=handles, loc=loc, title=title)


def save(fig, name):
    """Save a modular panel as PNG + PDF into panels/. Returns the PNG path."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    return FIG_DIR / f"{name}.png"
