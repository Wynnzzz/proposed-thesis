"""Visualisasi frontier (proyeksi 2D dari tiga objektif). Titik frontier diberi garis tepi."""
from __future__ import annotations
from typing import Dict, Sequence

COLORS = {"A3": "#1f77b4", "A1": "#ff7f0e", "A2": "#2ca02c", "B1": "#7f7f7f", "B2": "#d62728",
          "B3": "#9467bd", "B4": "#8c564b", "B5": "#e377c2"}


def plot_frontier(rows: Sequence[Dict], front_ids: set, path: str, title: str, selected: Dict[str, str] | None = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = [("cost", "error", "Biaya rata-rata (USD/query)", "Error"),
             ("latency", "error", "Latensi rata-rata (s, composed estimasi)", "Error"),
             ("cost", "latency", "Biaya rata-rata (USD/query)", "Latensi rata-rata (s, composed estimasi)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (x, y, xl, yl) in zip(axes, pairs):
        for kind in sorted({r["kind"] for r in rows}):
            rs = [r for r in rows if r["kind"] == kind]
            ax.scatter([r[x] for r in rs], [r[y] for r in rs], s=18, alpha=0.55, c=COLORS.get(kind, "k"), label=kind)
        fr = [r for r in rows if r["policy_id"] in front_ids]
        ax.scatter([r[x] for r in fr], [r[y] for r in fr], s=60, facecolors="none", edgecolors="k", linewidths=1.0,
                   label="frontier 3-objektif")
        for name, pid in (selected or {}).items():
            r = next((r for r in rows if r["policy_id"] == pid), None)
            if r:
                ax.scatter([r[x]], [r[y]], marker="*", s=220, c="gold", edgecolors="k", zorder=5)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(title + " (replay dev; bintang = kebijakan terpilih)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
