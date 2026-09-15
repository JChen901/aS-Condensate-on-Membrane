"""Pooled two-sample KS tests for persistent-largest-cluster lifetime data.

This script is read-only with respect to the persistent main analysis. All KS
tables and figures are written below pooled_ks_test/four_main_systems/.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
OUT = ROOT / "four_main_systems"
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "hel_idp_persistent_ks_mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

SYSTEMS = (
    ("G1_Wcluster_alhx0", "all", "IDP50"),
    ("G2_Wcluster_alhx700", "all", "Hel50"),
    ("G3_Onmem_plus", "all", "IDP50-Mem"),
    ("G4_Onmem_wall", "IDP-IDP", "Hel10IDP40-Mem (IDP only)"),
)
VARIANTS = ("strict_sym", "strict_anti", "banded_sym", "banded_anti")
REGISTRY_CONTACTS = ("one", "three", "triplet")
ANY_VARIANT = "strict_sym"
NO_REGISTRY_VARIANT = "none"
ARIAL_TTF_PATH = "/data/jychen/package/Arial/arial.ttf"
CONTACT_LABELS = {"any": "Any BB", "one": r"$\geq$1", "three": r"$\geq$3", "triplet": "3 consecutive"}
VARIANT_LABELS = {
    "strict_sym": "strict par.",
    "strict_anti": "strict anti-par.",
    "banded_sym": "banded par.",
    "banded_anti": "banded anti-par.",
}


def load_pub_font() -> str:
    if Path(ARIAL_TTF_PATH).exists():
        fm.fontManager.addfont(ARIAL_TTF_PATH)
        return fm.FontProperties(fname=ARIAL_TTF_PATH).get_name()
    return "Arial"


def mm_to_inch(mm: float) -> float:
    return mm / 25.4


def configure_style() -> None:
    font = load_pub_font()
    mpl.rcParams.update({
        "font.family": font,
        "font.sans-serif": [font, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.4,
        "ytick.major.width": 1.4,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.unicode_minus": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.dpi": 600,
        "figure.dpi": 120,
    })


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benjamini_hochberg(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def pooled_samples(histograms: dict[tuple[str, str], pd.DataFrame], group: str, pair: str,
                   variant: str, contact: str) -> np.ndarray:
    table = histograms[(group, pair)]
    data = table[(table.pair_type == pair) & (table.variant == variant) & (table.contact == contact)]
    counts = data.groupby("lifetime_frames", sort=True)["count"].sum()
    if len(counts) == 0:
        return np.array([], dtype=np.int32)
    return np.repeat(counts.index.to_numpy(dtype=np.int32), counts.to_numpy(dtype=np.int64))


def build_tests() -> tuple[pd.DataFrame, dict[str, str]]:
    histograms = {}
    manifest = {}
    for group, pair, _label in SYSTEMS:
        path = DATA / group / "event_histograms.csv"
        histograms[(group, pair)] = pd.read_csv(path)
        manifest[str(path)] = sha256(path)

    rows = []
    test_specs = [("any", NO_REGISTRY_VARIANT, ANY_VARIANT)]
    test_specs.extend((contact, variant, variant) for contact in REGISTRY_CONTACTS for variant in VARIANTS)
    for contact, output_variant, data_variant in test_specs:
        samples = {
            label: pooled_samples(histograms, group, pair, data_variant, contact)
            for group, pair, label in SYSTEMS
        }
        for label_a, label_b in combinations(samples, 2):
            a, b = samples[label_a], samples[label_b]
            result = ks_2samp(a, b, alternative="two-sided", method="asymp")
            rows.append({
                "variant": output_variant,
                "contact": contact,
                "system_a": label_a,
                "system_b": label_b,
                "n_a": len(a),
                "n_b": len(b),
                "ks_D": float(result.statistic),
                "p_value": float(result.pvalue),
            })

    table = pd.DataFrame(rows)
    table["q_value_global_BH"] = benjamini_hochberg(table.p_value.to_numpy())
    table["significant_q_0_05"] = table.q_value_global_BH < .05
    return table, manifest


def draw_heatmap_panel(ax, table: pd.DataFrame, contact: str, variant: str, column: str,
                       cap: float | None, title: str, ylabel: str | None,
                       show_x: bool, show_y: bool):
    labels = [item[2] for item in SYSTEMS]
    tick_labels = ["IDP50", "Hel50", "IDP50-Mem", "Hel10IDP40-Mem\n(IDP only)"]
    matrix = np.full((len(labels), len(labels)), np.nan)
    sub = table[(table.contact == contact) & (table.variant == variant)]
    for _, row in sub.iterrows():
        i, j = labels.index(row.system_a), labels.index(row.system_b)
        matrix[i, j] = matrix[j, i] = row[column]
    np.fill_diagonal(matrix, 0)
    shown = np.minimum(matrix, cap) if cap is not None else matrix.copy()
    shown = shown.copy()
    shown[np.triu_indices(len(labels), 1)] = np.nan
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    image = ax.imshow(shown, vmin=0, vmax=cap, cmap=cmap)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i < j:
                continue
            if i == j:
                text = "-"
            elif column == "neg_log10_q":
                text = f"{matrix[i,j]:.1f}" if matrix[i,j] < cap else f">={cap:.0f}"
            else:
                text = f"{matrix[i,j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=5.4,
                    color="white" if shown[i,j] > (cap or 1) * .55 else "black")
    ax.set_xticks(range(len(labels)), tick_labels if show_x else [], rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), tick_labels if show_y else [], fontsize=7)
    ax.set_title(title, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=8)
    ax.tick_params(length=0)
    return image


def plot_heatmaps(table: pd.DataFrame, column: str, stem: str, label: str, cap: float | None = None) -> None:
    fig = plt.figure(figsize=(mm_to_inch(165), mm_to_inch(142)))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.15], hspace=.42)
    top_grid = outer[0].subgridspec(1, 3, width_ratios=[1.5, 1, 1.5])
    grid = outer[1].subgridspec(3, 4, hspace=.32, wspace=.20)
    axes_for_colorbar = []
    ax = fig.add_subplot(top_grid[0, 1])
    image = draw_heatmap_panel(ax, table, "any", NO_REGISTRY_VARIANT, column, cap,
                               "Any BB", "Any BB", True, True)
    axes_for_colorbar.append(ax)
    for row, contact in enumerate(REGISTRY_CONTACTS):
        for col, variant in enumerate(VARIANTS):
            ax = fig.add_subplot(grid[row, col])
            image = draw_heatmap_panel(
                ax, table, contact, variant, column, cap,
                VARIANT_LABELS[variant] if row == 0 else "",
                CONTACT_LABELS[contact] if col == 0 else None,
                row == len(REGISTRY_CONTACTS) - 1, col == 0,
            )
            axes_for_colorbar.append(ax)
    fig.subplots_adjust(left=.10, right=.87, bottom=.10, top=.965)
    cax = fig.add_axes([.895, .22, .018, .56])
    fig.colorbar(image, cax=cax, label=label)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    configure_style()
    for directory in (OUT / "tables", OUT / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    table, manifest = build_tests()
    table.to_csv(OUT / "tables" / "stats_pooled_ks_pairwise_tests.csv", index=False)
    display = table.copy()
    display["neg_log10_q"] = -np.log10(np.maximum(display.q_value_global_BH, 1e-300))
    display.to_csv(OUT / "tables" / "stats_pooled_ks_pairwise_tests_neglogq.csv", index=False)
    plot_heatmaps(display, "ks_D", "pooled_ks_D", r"$D_{KS}$")
    plot_heatmaps(display, "neg_log10_q", "pooled_ks_global_BH_q", r"$-\log_{10}$(global BH q)", cap=20)
    (OUT / "input_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "n_tests": len(table),
        "n_significant_global_BH_q_0_05": int(table.significant_q_0_05.sum()),
        "input_files": len(manifest),
        "input_files_unchanged": all(sha256(Path(path)) == digest for path, digest in manifest.items()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
