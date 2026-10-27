"""Shared figure style: one fixed color per AAMI class across all figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_COLORS = {
    "N": "#2a78d6",  # blue
    "S": "#eb6834",  # orange
    "V": "#1baf7a",  # aqua
    "F": "#eda100",  # yellow
    "Q": "#e87ba4",  # magenta
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.2, "legend.frameon": False,
    })
