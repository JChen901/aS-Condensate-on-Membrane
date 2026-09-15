from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "registry_window_sensitivity_mpl"),
)

import matplotlib
matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"

ARIAL_TTF_PATH = "/data/jychen/package/Arial/arial.ttf"
USE_ARIAL_TTF = True
_PUB_FONT_NAME = None

FIG_WIDTH_MM = {
    "single": 100,
    "double": 200,
}
PANEL_LETTER_FONTSIZE = 24
BAR_EDGE_LINEWIDTH = 0.8

WINDOWS = (4, 6, 8, 10)
REPS = (1, 2, 3)

GROUP_ORDER = [
    "G1_Wcluster_alhx0",
    "G2_Wcluster_alhx700",
    "G3_Onmem_plus",
    "G4_Onmem_wall",
]

# Match plot_nac_contact_definition_figures.py exactly, including true subscripts.
GROUP_LABELS = {
    "G1_Wcluster_alhx0": r"$\mathrm{IDP}_{50}$",
    "G2_Wcluster_alhx700": r"$\mathrm{Hel}_{50}$",
    "G3_Onmem_plus": r"$\mathrm{IDP}_{50}$-Mem",
    "G4_Onmem_wall": r"$\mathrm{Hel}_{10}\mathrm{IDP}_{40}$-Mem (IDP only)",
}

GROUP_COLORS = {
    "G1_Wcluster_alhx0": "#4C78A8",
    "G2_Wcluster_alhx700": "#F58518",
    "G3_Onmem_plus": "#54A24B",
    "G4_Onmem_wall": "#B279A2",
}

PAIR_TYPES = {
    "G1_Wcluster_alhx0": "all",
    "G2_Wcluster_alhx700": "all",
    "G3_Onmem_plus": "all",
    "G4_Onmem_wall": "IDP-IDP",
}

CONTACT_LABELS = {
    "one": r"$\geq$1",
    "three": r"$\geq$3",
    "triplet": "3 consec.",
}

VARIANT_LABELS = {
    "banded_sym": "banded par.",
    "banded_anti": "banded anti-par.",
}

METRIC_LABELS = {
    "t_01_ps": r"$\mathrm{T}_{\mathrm{S}}(0.1)$ (ns)",
    "t_1e_ps": r"$\mathrm{T}_{\mathrm{S}}(1/e)$ (ns)",
}


def load_pub_font():
    global _PUB_FONT_NAME
    if _PUB_FONT_NAME is not None:
        return _PUB_FONT_NAME

    if USE_ARIAL_TTF and os.path.exists(ARIAL_TTF_PATH):
        fm.fontManager.addfont(ARIAL_TTF_PATH)
        _PUB_FONT_NAME = fm.FontProperties(fname=ARIAL_TTF_PATH).get_name()
    else:
        _PUB_FONT_NAME = "DejaVu Sans"
    return _PUB_FONT_NAME


def mm_to_inch(mm):
    return mm / 25.4


def configure_style():
    """Same publication style as plot_nac_contact_definition_figures.py."""
    pub_font = load_pub_font()
    mpl.rcParams.update(
        {
            "font.family": pub_font,
            "font.sans-serif": [pub_font, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": pub_font,
            "mathtext.it": pub_font,
            "mathtext.bf": pub_font,
            "mathtext.default": "regular",
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.minor.width": 1.1,
            "ytick.minor.width": 1.1,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.size": 2.8,
            "ytick.minor.size": 2.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 1.7,
            "patch.linewidth": 1.3,
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            "legend.handletextpad": 0.4,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "axes.unicode_minus": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def new_subplots_mm(nrows, ncols, width_mm, height_mm, **kwargs):
    return plt.subplots(
        nrows,
        ncols,
        figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)),
        **kwargs,
    )


def save_figure(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved {path}")
    plt.close(fig)


def normalize(table):
    """Split internal variant names such as banded_sym_w8."""
    if "registry_window" in table.columns:
        return table.copy()

    table = table.copy()
    parts = table["variant"].str.rsplit("_w", n=1, expand=True)
    table["variant"] = parts[0]
    table["registry_window"] = parts[1].astype(int)
    return table


def load_summary_table():
    """T_S extracted from the mean survival curve, as in the established analysis."""
    tables = []
    for group in GROUP_ORDER:
        path = ROOT / "data" / group / "lifetime_summary.csv"
        table = normalize(pd.read_csv(path))
        table["group"] = group
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def load_replicate_table():
    """Independent rep1/rep2/rep3 T_S values from trajectory_summary.csv."""
    tables = []
    for group in GROUP_ORDER:
        path = ROOT / "data" / group / "trajectory_summary.csv"
        table = normalize(pd.read_csv(path))
        table["group"] = group
        tables.append(table)

    table = pd.concat(tables, ignore_index=True)
    available_reps = set(table["replicate"].astype(int).unique())
    missing = set(REPS) - available_reps
    if missing:
        raise ValueError(f"Missing replicate(s) in trajectory_summary.csv: {sorted(missing)}")
    return table


def get_system_subset(table, group, variant, contact):
    pair_type = PAIR_TYPES[group]
    return table[
        (table["group"] == group)
        & (table["pair_type"] == pair_type)
        & (table["variant"] == variant)
        & (table["contact"] == contact)
    ].copy()


def nice_linear_upper(value):
    """Same y-limit helper logic as the reference plotting script."""
    if not np.isfinite(value) or value <= 0:
        return 1.0
    value *= 1.08
    if value <= 10:
        step = 1.0
    elif value <= 20:
        step = 2.0
    elif value <= 50:
        step = 5.0
    else:
        step = 10.0
    return np.ceil(value / step) * step


def draw_panel(ax, summary, replicates, metric, variant, contact, ylabel=None, show_xticklabels=True):
    """Grouped bars + hollow replicate points, matching the reference figure logic."""
    x = np.arange(len(WINDOWS), dtype=float)

    # Exactly the same grouped-bar geometry used in draw_definition_group_bars().
    total_width = 0.78
    bar_width = total_width / len(GROUP_ORDER)
    offsets = (np.arange(len(GROUP_ORDER)) - (len(GROUP_ORDER) - 1) / 2.0) * bar_width

    for offset, group in zip(offsets, GROUP_ORDER):
        # ---- bar heights: established summary T_S from the mean survival curve ----
        sum_sub = get_system_subset(summary, group, variant, contact)
        sum_sub = sum_sub[sum_sub["registry_window"].isin(WINDOWS)].sort_values("registry_window")

        if len(sum_sub) != len(WINDOWS):
            raise ValueError(
                f"Unexpected summary coverage for {group}, {variant}, {contact}: "
                f"got {len(sum_sub)} rows, expected {len(WINDOWS)}."
            )
        if tuple(sum_sub["registry_window"].astype(int)) != WINDOWS:
            raise ValueError(
                f"Wrong registry-window order for {group}, {variant}, {contact}: "
                f"{sum_sub['registry_window'].tolist()}"
            )

        values = sum_sub[metric].to_numpy(dtype=float) / 1000.0

        ax.bar(
            x + offset,
            values,
            width=bar_width * 0.82,
            color=GROUP_COLORS[group],
            edgecolor="black",
            linewidth=BAR_EDGE_LINEWIDTH,
            label=GROUP_LABELS[group],
            zorder=3,
        )

        # ---- individual replicate points: white fill + black edge, as in reference ----
        rep_sub = get_system_subset(replicates, group, variant, contact)
        rep_sub = rep_sub[rep_sub["registry_window"].isin(WINDOWS)].copy()

        expected_rows = len(WINDOWS) * len(REPS)
        if len(rep_sub) != expected_rows:
            raise ValueError(
                f"Unexpected replicate coverage for {group}, {variant}, {contact}: "
                f"got {len(rep_sub)} rows, expected {expected_rows}."
            )

        errors = []
        for idx, window in enumerate(WINDOWS):
            reps = rep_sub[rep_sub["registry_window"] == window].sort_values("replicate")
            if tuple(reps["replicate"].astype(int)) != REPS:
                raise ValueError(
                    f"Incomplete replicate coverage for {group}, {variant}, {contact}, "
                    f"window={window}: {reps['replicate'].tolist()}"
                )

            y = reps[metric].to_numpy(dtype=float) / 1000.0
            y = y[np.isfinite(y)]
            if len(y) == 0:
                errors.append(0.0)
                continue

            if len(y) == 1:
                jitter = np.array([0.0])
                errors.append(0.0)
            else:
                jitter = np.linspace(-bar_width * 0.24, bar_width * 0.24, len(y))
                errors.append(float(np.std(y, ddof=1)))

            ax.scatter(
                np.full(len(y), x[idx] + offset) + jitter,
                y,
                s=13,
                marker="o",
                facecolors="white",
                edgecolors="black",
                linewidths=0.55,
                zorder=5,
                label="_nolegend_",
            )

        ax.errorbar(
            x + offset,
            values,
            yerr=np.asarray(errors, dtype=float),
            fmt="none",
            ecolor="black",
            elinewidth=BAR_EDGE_LINEWIDTH,
            capsize=1.6,
            capthick=BAR_EDGE_LINEWIDTH,
            zorder=4,
        )

    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels([str(w) for w in WINDOWS], rotation=0, ha="center")
    else:
        ax.set_xticklabels([])

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(False)
    ax.tick_params(top=False, right=False)
    ax.minorticks_off()


def set_common_row_ylim(axes, summary, replicates, metric, contact):
    """Use one y-range for parallel and anti-parallel panels in the same row."""
    values = []
    for variant in ("banded_sym", "banded_anti"):
        for group in GROUP_ORDER:
            s = get_system_subset(summary, group, variant, contact)
            s = s[s["registry_window"].isin(WINDOWS)]
            values.extend((s[metric].to_numpy(dtype=float) / 1000.0).tolist())

            r = get_system_subset(replicates, group, variant, contact)
            r = r[r["registry_window"].isin(WINDOWS)]
            values.extend((r[metric].to_numpy(dtype=float) / 1000.0).tolist())

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    ymax = nice_linear_upper(np.nanmax(values))
    for ax in np.asarray(axes).ravel():
        ax.set_ylim(0, ymax)


def plot(metric):
    configure_style()

    summary = load_summary_table()
    replicates = load_replicate_table()

    # Same 200-mm double-column philosophy as the user's reference script.
    fig, axes = new_subplots_mm(
        3,
        2,
        width_mm=FIG_WIDTH_MM["double"],
        height_mm=175,
        sharex=True,
        sharey=False,
    )

    contacts = ("one", "three", "triplet")
    variants = ("banded_sym", "banded_anti")

    for row, contact in enumerate(contacts):
        for col, variant in enumerate(variants):
            ax = axes[row, col]
            draw_panel(
                ax,
                summary,
                replicates,
                metric,
                variant,
                contact,
                ylabel=METRIC_LABELS[metric] if col == 0 else None,
                show_xticklabels=(row == len(contacts) - 1),
            )

            # Same title treatment as the reference multi-panel figures.
            if row == 0:
                ax.set_title(VARIANT_LABELS[variant])

            # Same small white-box panel descriptor used in plot_metric_grid().
            ax.text(
                0.03,
                0.94,
                CONTACT_LABELS[contact],
                transform=ax.transAxes,
                fontsize=mpl.rcParams["legend.fontsize"],
                ha="left",
                va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.35),
                zorder=20,
            )

        set_common_row_ylim(
            axes[row, :],
            summary,
            replicates,
            metric,
            contact,
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Registry window")

    # Legend comes from bars only; replicate circles stay out of the legend.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Keep legend labels on a single line.
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )

    # Spacing adapted from the user's plot_metric_grid() to a 3x2 layout.
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.10,
        top=0.90,
        hspace=0.44,
        wspace=0.24,
    )

    save_figure(
        fig,
        "registry_window_sensitivity_" + metric.replace("_ps", ""),
    )


if __name__ == "__main__":
    plot("t_01_ps")
    plot("t_1e_ps")
