from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name == "fig":
    BASE_DIR = SCRIPT_DIR.parent
    FIG_DIR = SCRIPT_DIR
else:
    BASE_DIR = SCRIPT_DIR
    FIG_DIR = BASE_DIR / "fig"
ARIAL_TTF_PATH = "/data/jychen/package/Arial/arial.ttf"
USE_ARIAL_TTF = True
_PUB_FONT_NAME = None

FIG_WIDTH_MM = {
    "single": 100,
    "double": 200,
}
FIG_PRESETS = {
    "single_medium": ("single", 75),
    "double_medium": ("double", 100),
}
PANEL_LETTER_FONTSIZE = 24

GROUP_ORDER = [
    "G1_Wcluster_alhx0",
    "G2_Wcluster_alhx700",
    "G3_Onmem_plus",
    "G4_Onmem_wall",
]
GROUP_LABELS = {
    "G1_Wcluster_alhx0": r"$\mathrm{IDP}_{50}$",
    "G2_Wcluster_alhx700": r"$\mathrm{Hel}_{50}$",
    "G3_Onmem_plus": r"$\mathrm{IDP}_{50}$-Mem",
    "G4_Onmem_wall": r"$\mathrm{Hel}_{10}\mathrm{IDP}_{40}$-Mem",
}
GROUP_COLORS = {
    "G1_Wcluster_alhx0": "#4C78A8",
    "G2_Wcluster_alhx700": "#F58518",
    "G3_Onmem_plus": "#54A24B",
    "G4_Onmem_wall": "#B279A2",
}
_WARNED_EMPTY_REPLICATES = set()
BAR_EDGE_LINEWIDTH = 0.8

VARIANTS = OrderedDict(
    [
        (
            "strict_sym",
            {
                "label": "corr.",
                "folder": "NAC_contact_relaxation_time",
                "color": "#2F6DAE",
            },
        ),
        (
            "strict_anti",
            {
                "label": "anti.",
                "folder": "NAC_antisymmetric_contact_relaxation_time",
                "color": "#C84C47",
            },
        ),
        (
            "banded_sym",
            {
                "label": "band sym.",
                "folder": "NAC_banded_symmetric_contact_relaxation_time",
                "color": "#269C7F",
            },
        ),
        (
            "banded_anti",
            {
                "label": "band anti.",
                "folder": "NAC_banded_antisymmetric_contact_relaxation_time",
                "color": "#D98C24",
            },
        ),
    ]
)

CONTACT_DEFS = OrderedDict(
    [
        (
            "any",
            {
                "label": "Any BB-BB",
                "short_label": "Any BB-BB",
                "panel_labels": {
                    "strict_sym": "Any BB-BB",
                    "strict_anti": "Any BB-BB",
                    "banded_sym": "Any BB-BB",
                    "banded_anti": "Any BB-BB",
                },
                "legend_labels": {
                    "strict_sym": "Any BB-BB",
                    "strict_anti": "Any BB-BB",
                    "banded_sym": "Any BB-BB",
                    "banded_anti": "Any BB-BB",
                },
                "modes": {
                    "strict_sym": "any_nac_to_nac_bb",
                    "strict_anti": "any_nac_to_nac_bb",
                    "banded_sym": "any_nac_to_nac_bb",
                    "banded_anti": "any_nac_to_nac_bb",
                },
            },
        ),
        (
            "three",
            {
                "label": r"$\geq$3 corr./anti./banded",
                "short_label": r"$\geq$3",
                "panel_labels": {
                    "strict_sym": r"$\geq$3 corr.",
                    "strict_anti": r"$\geq$3 anti.",
                    "banded_sym": r"$\geq$3 band sym.",
                    "banded_anti": r"$\geq$3 band anti.",
                },
                "legend_labels": {
                    "strict_sym": r"$\geq$3 corr.",
                    "strict_anti": r"$\geq$3 anti.",
                    "banded_sym": r"$\geq$3 band sym.",
                    "banded_anti": r"$\geq$3 band anti.",
                },
                "modes": {
                    "strict_sym": "three_corresponding_nac_bb",
                    "strict_anti": "three_antisymmetric_nac_bb",
                    "banded_sym": "three_symmetric_nac_bb",
                    "banded_anti": "three_antisymmetric_nac_bb",
                },
            },
        ),
        (
            "one",
            {
                "label": r"$\geq$1 corr./anti./banded",
                "short_label": r"$\geq$1",
                "panel_labels": {
                    "strict_sym": r"$\geq$1 corr.",
                    "strict_anti": r"$\geq$1 anti.",
                    "banded_sym": r"$\geq$1 band sym.",
                    "banded_anti": r"$\geq$1 band anti.",
                },
                "legend_labels": {
                    "strict_sym": r"$\geq$1 corr.",
                    "strict_anti": r"$\geq$1 anti.",
                    "banded_sym": r"$\geq$1 band sym.",
                    "banded_anti": r"$\geq$1 band anti.",
                },
                "modes": {
                    "strict_sym": "corresponding_nac_bb",
                    "strict_anti": "antisymmetric_nac_bb",
                    "banded_sym": "symmetric_nac_bb",
                    "banded_anti": "antisymmetric_nac_bb",
                },
            },
        ),
        (
            "triplet",
            {
                "label": "3 consec. corr./anti./banded",
                "short_label": "Triplet",
                "panel_labels": {
                    "strict_sym": "3 consec.",
                    "strict_anti": "3 anti consec.",
                    "banded_sym": "3 band sym consec.",
                    "banded_anti": "3 band anti consec.",
                },
                "legend_labels": {
                    "strict_sym": "3 consec.",
                    "strict_anti": "3 anti consec.",
                    "banded_sym": "3 band sym consec.",
                    "banded_anti": "3 band anti consec.",
                },
                "modes": {
                    "strict_sym": "triplet_corresponding_nac_bb",
                    "strict_anti": "triplet_antisymmetric_nac_bb",
                    "banded_sym": "triplet_symmetric_nac_bb",
                    "banded_anti": "triplet_antisymmetric_nac_bb",
                },
            },
        ),
    ]
)


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


def figsize_from_preset(preset, height_mm=None):
    mode, preset_height_mm = FIG_PRESETS[preset]
    width_mm = FIG_WIDTH_MM[mode]
    if height_mm is None:
        height_mm = preset_height_mm
    return mm_to_inch(width_mm), mm_to_inch(height_mm)


def new_subplots(nrows, ncols, preset="double_medium", height_mm=None, **kwargs):
    return plt.subplots(
        nrows,
        ncols,
        figsize=figsize_from_preset(preset, height_mm=height_mm),
        **kwargs,
    )


def new_subplots_mm(nrows, ncols, width_mm, height_mm, **kwargs):
    return plt.subplots(
        nrows,
        ncols,
        figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)),
        **kwargs,
    )


def save_figure(fig, stem, tight=True):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{stem}.png"
    if tight:
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)


def metric_path(variant_key, kind, threshold_key):
    folder = BASE_DIR / VARIANTS[variant_key]["folder"]
    if kind == "survival":
        return folder / "nac_contact_outputs" / f"nac_contact_survival_{threshold_key}_summary.csv"
    if kind == "acf":
        return folder / "nac_contact_acf_outputs" / f"nac_contact_acf_{threshold_key}_summary.csv"
    raise ValueError(f"Unknown kind: {kind}")


def load_metric_rows(contact_def_keys, variant_keys, kind, threshold_key):
    rows = []
    for contact_def_key in contact_def_keys:
        contact_def = CONTACT_DEFS[contact_def_key]
        for variant_key in variant_keys:
            variant = VARIANTS[variant_key]
            mode = contact_def["modes"][variant_key]
            path = metric_path(variant_key, kind, threshold_key)
            df = pd.read_csv(path)
            sub = df[df["contact_mode"] == mode].copy()
            if sub.empty:
                raise ValueError(f"No rows for contact_mode={mode!r} in {path}")
            sub["contact_def_key"] = contact_def_key
            sub["contact_def_label"] = contact_def["label"]
            sub["variant_key"] = variant_key
            sub["variant_label"] = variant["label"]
            sub["variant_folder"] = variant["folder"]
            sub["source_csv"] = str(path.relative_to(BASE_DIR))
            rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def load_replicate_metric_rows(contact_def_keys, variant_keys, kind, threshold_key):
    rows = []
    threshold_col = f"{threshold_key}_ps"
    for contact_def_key in contact_def_keys:
        for variant_key in variant_keys:
            data_path, _value_col, _std_col = find_data_file(variant_key, contact_def_key, kind)
            if kind == "survival":
                traj_name = data_path.name.replace("_survival_data.csv", "_trajectory_summary.csv")
            else:
                traj_name = data_path.name.replace("_data.csv", "_trajectory_summary.csv")
            traj_path = data_path.with_name(traj_name)
            if not traj_path.exists():
                continue
            header = pd.read_csv(traj_path, nrows=0)
            if threshold_col not in header.columns:
                continue
            usecols = ["group", "trajectory", threshold_col]
            df = pd.read_csv(traj_path, usecols=usecols)
            df = df[pd.notna(df[threshold_col])].copy()
            if df.empty:
                continue
            df["time_ns"] = df[threshold_col].astype(float) / 1000.0
            df = df[np.isfinite(df["time_ns"])]
            if df.empty:
                continue
            df["contact_def_key"] = contact_def_key
            df["variant_key"] = variant_key
            df["source_csv"] = str(traj_path.relative_to(BASE_DIR))
            rows.append(df)
    if not rows:
        warning_key = (kind, threshold_key)
        if warning_key not in _WARNED_EMPTY_REPLICATES:
            print(
                f"Warning: no per-trajectory {kind} {threshold_key} values found in "
                f"trajectory_summary CSVs; hollow replicate circles will be skipped. "
                f"Rerun the analysis scripts to refresh {threshold_col}."
            )
            _WARNED_EMPTY_REPLICATES.add(warning_key)
        return pd.DataFrame(
            columns=["group", "trajectory", "time_ns", "contact_def_key", "variant_key", "source_csv"]
        )
    return pd.concat(rows, ignore_index=True)


def threshold_axis_label(kind, threshold_label):
    if kind == "survival":
        return rf"$\mathrm{{T}}_{{\mathrm{{S}}}}({threshold_label})$ (ns)"
    return rf"$\mathrm{{T}}_{{\mathrm{{ACF}}}}({threshold_label})$ (ns)"


def decay_axis_labels(kind):
    if kind == "acf":
        return r"$\mathrm{ACF}(\mathrm{t})$", r"$t$ (ns)"
    return r"$\mathrm{S}(\mathrm{t})$", r"$t$ (ns)"


def contact_panel_label(contact_def_key, variant_key):
    return CONTACT_DEFS[contact_def_key]["panel_labels"][variant_key]


def contact_legend_label(contact_def_key, variant_key):
    if contact_def_key == "any":
        return VARIANTS[variant_key]["label"]
    return CONTACT_DEFS[contact_def_key]["legend_labels"][variant_key]


def definition_axis_variant_keys(contact_def_key):
    if contact_def_key == "any":
        return ("strict_sym",)
    return tuple(VARIANTS.keys())


def definition_axis_label(contact_def_key, variant_key):
    if contact_def_key == "any":
        return CONTACT_DEFS[contact_def_key]["short_label"]
    return contact_legend_label(contact_def_key, variant_key)


def find_data_file(variant_key, contact_def_key, kind):
    folder = BASE_DIR / VARIANTS[variant_key]["folder"]
    mode = CONTACT_DEFS[contact_def_key]["modes"][variant_key]
    if kind == "survival":
        root = folder / "nac_contact_outputs"
        pattern = "*_survival_data.csv"
        value_col = "survival_mean"
    elif kind == "acf":
        root = folder / "nac_contact_acf_outputs"
        pattern = "*_acf_data.csv"
        value_col = "acf_mean"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    for path in sorted(root.glob(f"*/{pattern}")):
        header = pd.read_csv(path, nrows=1)
        if "contact_mode" in header.columns and header.loc[0, "contact_mode"] == mode:
            std_col = value_col.replace("_mean", "_std")
            return path, value_col, std_col
    raise FileNotFoundError(f"No {kind} data file for contact_mode={mode!r} under {root}")


def load_decay_data(variant_key, contact_def_key, kind):
    path, value_col, std_col = find_data_file(variant_key, contact_def_key, kind)
    columns = ["group", "time_ps", value_col]
    available = set(pd.read_csv(path, nrows=0).columns)
    if std_col in available:
        columns.append(std_col)
    df = pd.read_csv(path, usecols=columns)
    df["time_ns"] = df["time_ps"] / 1000.0
    df["source_csv"] = str(path.relative_to(BASE_DIR))
    return df, value_col, std_col


def value_error_for(df, contact_def_key, variant_key, group):
    sub = df[
        (df["contact_def_key"] == contact_def_key)
        & (df["variant_key"] == variant_key)
        & (df["group"] == group)
    ]
    if sub.empty:
        return np.nan, 0.0
    row = sub.iloc[0]
    if not bool(row.get("threshold_reached", True)):
        return np.nan, 0.0
    value = float(row["time_ns"]) if pd.notna(row["time_ns"]) else np.nan
    err = float(row["err_sym_ns"]) if "err_sym_ns" in row and pd.notna(row["err_sym_ns"]) else 0.0
    if np.isfinite(value) and value > 0:
        err = min(err, value * 0.8)
    else:
        err = 0.0
    return value, err


def nice_linear_upper(value):
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


def nice_log_upper(value):
    if not np.isfinite(value) or value <= 0:
        return 10.0
    value *= 1.15
    exponent = np.floor(np.log10(value))
    base = value / (10**exponent)
    if base <= 2:
        return 2 * (10**exponent)
    if base <= 5:
        return 5 * (10**exponent)
    return 10 ** (exponent + 1)


def nice_log_lower(value):
    if not np.isfinite(value) or value <= 0:
        return 0.5
    value *= 0.8
    exponent = np.floor(np.log10(value))
    base = value / (10**exponent)
    if base >= 5:
        return 5 * (10**exponent)
    if base >= 2:
        return 2 * (10**exponent)
    return 10**exponent


def metric_extent_values(df, contact_def_keys, variant_keys, replicate_df=None):
    values = []
    for contact_def_key in contact_def_keys:
        for variant_key in definition_axis_variant_keys(contact_def_key):
            if variant_key not in variant_keys:
                continue
            for group in GROUP_ORDER:
                sub = df[
                    (df["contact_def_key"] == contact_def_key)
                    & (df["variant_key"] == variant_key)
                    & (df["group"] == group)
                ]
                for _idx, row in sub.iterrows():
                    if not bool(row.get("threshold_reached", True)):
                        continue
                    value = float(row["time_ns"]) if pd.notna(row["time_ns"]) else np.nan
                    err = (
                        float(row["err_sym_ns"])
                        if "err_sym_ns" in row and pd.notna(row["err_sym_ns"])
                        else 0.0
                    )
                    if np.isfinite(value):
                        values.append(value + max(err, 0.0))
    if replicate_df is not None and not replicate_df.empty:
        sub = replicate_df[
            replicate_df["contact_def_key"].isin(contact_def_keys)
            & replicate_df["variant_key"].isin(variant_keys)
        ]
        values.extend(sub["time_ns"].to_numpy(dtype=float))
    return np.asarray(values, dtype=float)


def set_common_metric_ylim(axes, df, contact_def_keys, variant_keys, replicate_df=None, log_y=False):
    values = metric_extent_values(df, contact_def_keys, variant_keys, replicate_df=replicate_df)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    axes = np.asarray(axes).ravel()
    if log_y:
        positive = values[values > 0]
        if positive.size == 0:
            return
        ymin = nice_log_lower(np.nanmin(positive))
        ymax = nice_log_upper(np.nanmax(positive))
        for ax in axes:
            ax.set_ylim(ymin, ymax)
    else:
        ymax = nice_linear_upper(np.nanmax(values))
        for ax in axes:
            ax.set_ylim(0, ymax)


def draw_definition_group_bars(
    ax,
    df,
    contact_def_key,
    ylabel=None,
    log_y=False,
    show_xticklabels=True,
    replicate_df=None,
):
    x_variant_keys = definition_axis_variant_keys(contact_def_key)
    x = np.arange(len(x_variant_keys), dtype=float)
    total_width = 0.78
    bar_width = total_width / len(GROUP_ORDER)
    offsets = (np.arange(len(GROUP_ORDER)) - (len(GROUP_ORDER) - 1) / 2.0) * bar_width

    for offset, group in zip(offsets, GROUP_ORDER):
        values = []
        errors = []
        for variant_key in x_variant_keys:
            value, err = value_error_for(df, contact_def_key, variant_key, group)
            values.append(value)
            errors.append(err)
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
        ax.errorbar(
            x + offset,
            values,
            yerr=errors,
            fmt="none",
            ecolor="black",
            elinewidth=BAR_EDGE_LINEWIDTH,
            capsize=1.6,
            capthick=BAR_EDGE_LINEWIDTH,
            zorder=4,
        )
        if replicate_df is not None and not replicate_df.empty:
            for idx, variant_key in enumerate(x_variant_keys):
                reps = replicate_df[
                    (replicate_df["contact_def_key"] == contact_def_key)
                    & (replicate_df["variant_key"] == variant_key)
                    & (replicate_df["group"] == group)
                ]
                if reps.empty:
                    continue
                y = reps["time_ns"].to_numpy(dtype=float)
                y = y[np.isfinite(y)]
                if len(y) == 0:
                    continue
                if len(y) == 1:
                    jitter = np.array([0.0])
                else:
                    jitter = np.linspace(-bar_width * 0.24, bar_width * 0.24, len(y))
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

    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(
            [definition_axis_label(contact_def_key, variant_key) for variant_key in x_variant_keys],
            rotation=30,
            ha="right",
            rotation_mode="anchor",
        )
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(False)
    ax.tick_params(top=False, right=False)
    ax.minorticks_off()


def plot_main_triplet_t01():
    contact_def_keys = ["triplet"]
    variant_keys = list(VARIANTS)
    df = load_metric_rows(contact_def_keys, variant_keys, "survival", "t_01")
    replicate_df = load_replicate_metric_rows(contact_def_keys, variant_keys, "survival", "t_01")
    fig, ax = new_subplots_mm(1, 1, width_mm=90, height_mm=100)
    draw_definition_group_bars(
        ax,
        df,
        "triplet",
        ylabel=threshold_axis_label("survival", "0.1"),
        replicate_df=replicate_df,
    )
    ax.set_ylim(0, 8)
    ax.set_yticks([2, 4, 6, 8])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=2,
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.75,
        facecolor="white",
        edgecolor="none",
        handlelength=mpl.rcParams["legend.handlelength"],
    )
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.265, top=0.89)
    save_figure(fig, "main_nac_triplet_lifetime_t01_strict_banded", tight=False)


def plot_metric_grid(kind, threshold_key, ylabel, stem, log_y=False):
    contact_def_keys = list(CONTACT_DEFS)
    variant_keys = list(VARIANTS)
    df = load_metric_rows(contact_def_keys, variant_keys, kind, threshold_key)
    replicate_df = load_replicate_metric_rows(contact_def_keys, variant_keys, kind, threshold_key)
    fig, axes = new_subplots(2, 2, preset="double_medium", height_mm=125, sharey=False)
    axes = axes.ravel()
    for idx, (ax, contact_def_key) in enumerate(zip(axes, contact_def_keys)):
        draw_definition_group_bars(
            ax,
            df,
            contact_def_key,
            ylabel=ylabel if idx in (0, 2) else None,
            log_y=log_y,
            replicate_df=replicate_df,
        )
        ax.text(
            -0.14,
            1.00,
            chr(65 + idx),
            transform=ax.transAxes,
            fontsize=PANEL_LETTER_FONTSIZE,
            ha="right",
            va="top",
            clip_on=False,
        )
    set_common_metric_ylim(
        axes,
        df,
        contact_def_keys,
        variant_keys,
        replicate_df=replicate_df,
        log_y=log_y,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.13, top=0.89, hspace=0.42, wspace=0.22)
    save_figure(fig, stem)


def plot_decay_grid(kind, stem, x_max_ns):
    contact_def_keys = list(CONTACT_DEFS)
    variant_keys = list(VARIANTS)
    ylabel, xlabel = decay_axis_labels(kind)
    fig, axes = new_subplots(
        len(contact_def_keys),
        len(variant_keys),
        preset="double_medium",
        height_mm=180,
        sharex=True,
        sharey=True,
    )
    for row_idx, contact_def_key in enumerate(contact_def_keys):
        for col_idx, variant_key in enumerate(variant_keys):
            ax = axes[row_idx, col_idx]
            df, value_col, _std_col = load_decay_data(variant_key, contact_def_key, kind)
            df = df[df["time_ns"] <= x_max_ns].copy()
            for group in GROUP_ORDER:
                sub = df[df["group"] == group]
                if sub.empty:
                    continue
                ax.plot(
                    sub["time_ns"],
                    sub[value_col],
                    color=GROUP_COLORS[group],
                    linewidth=1.0,
                    label=GROUP_LABELS[group],
                )
            ax.axhline(1.0 / np.e, color="#2ca02c", linewidth=0.70, linestyle="--", alpha=0.85)
            ax.axhline(0.1, color="#9467bd", linewidth=0.70, linestyle="--", alpha=0.85)
            ax.text(
                0.98,
                1.0 / np.e,
                "1/e",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                color="#2ca02c",
                fontsize=mpl.rcParams["legend.fontsize"],
            )
            ax.text(
                0.98,
                0.1,
                "0.1",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                color="#9467bd",
                fontsize=mpl.rcParams["legend.fontsize"],
            )
            if kind == "acf":
                ax.axhline(0.0, color="#9A9A9A", linewidth=0.52)
                ax.set_ylim(-0.18, 1.03)
            else:
                ax.set_ylim(-0.02, 1.03)
            ax.set_xlim(0, x_max_ns)
            ax.set_xticks([0, x_max_ns / 2, x_max_ns])
            labels = ax.get_xticklabels()
            if labels:
                labels[0].set_ha("left")
                labels[-1].set_ha("right")
            ax.grid(False)
            ax.tick_params(top=False, right=False)
            ax.minorticks_off()
            ax.text(
                0.04,
                0.92,
                contact_panel_label(contact_def_key, variant_key),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=mpl.rcParams["legend.fontsize"],
            )
            if row_idx == 0:
                ax.set_title(VARIANTS[variant_key]["label"])
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == len(contact_def_keys) - 1:
                ax.set_xlabel(xlabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.91, hspace=0.34, wspace=0.16)
    save_figure(fig, stem)


def plot_acf_metric_grid():
    contact_def_keys = list(CONTACT_DEFS)
    variant_keys = list(VARIANTS)
    loaded = {
        "t_01": (
            load_metric_rows(contact_def_keys, variant_keys, "acf", "t_01"),
            load_replicate_metric_rows(contact_def_keys, variant_keys, "acf", "t_01"),
        ),
        "t_1e": (
            load_metric_rows(contact_def_keys, variant_keys, "acf", "t_1e"),
            load_replicate_metric_rows(contact_def_keys, variant_keys, "acf", "t_1e"),
        ),
    }
    rows = [
        ("t_01", threshold_axis_label("acf", "0.1"), "0.1"),
        ("t_1e", threshold_axis_label("acf", "1/e"), "1/e"),
    ]
    fig, axes = new_subplots(
        len(contact_def_keys),
        len(rows),
        preset="double_medium",
        height_mm=215,
        sharex=False,
    )
    for row_idx, contact_def_key in enumerate(contact_def_keys):
        for col_idx, (threshold_key, ylabel, _threshold_label) in enumerate(rows):
            df, replicate_df = loaded[threshold_key]
            ax = axes[row_idx, col_idx]
            draw_definition_group_bars(
                ax,
                df,
                contact_def_key,
                ylabel=ylabel,
                log_y=True,
                replicate_df=replicate_df,
            )
    combined_df = pd.concat([loaded[threshold_key][0] for threshold_key, _ylabel, _label in rows], ignore_index=True)
    combined_replicate_df = pd.concat(
        [loaded[threshold_key][1] for threshold_key, _ylabel, _label in rows],
        ignore_index=True,
    )
    set_common_metric_ylim(
        axes,
        combined_df,
        contact_def_keys,
        variant_keys,
        replicate_df=combined_replicate_df,
        log_y=True,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.925, hspace=0.54, wspace=0.34)
    save_figure(fig, "si_nac_all_contact_definitions_acf_metrics")


def main():
    configure_style()
    plot_main_triplet_t01()
    plot_metric_grid(
        "survival",
        "t_01",
        threshold_axis_label("survival", "0.1"),
        "si_nac_all_contact_definitions_lifetime_t01",
    )
    plot_metric_grid(
        "survival",
        "t_1e",
        threshold_axis_label("survival", "1/e"),
        "si_nac_all_contact_definitions_lifetime_t1e",
    )
    plot_decay_grid(
        "survival",
        "si_nac_all_contact_definitions_survival_decay",
        x_max_ns=20.0,
    )
    plot_acf_metric_grid()
    plot_decay_grid(
        "acf",
        "si_nac_all_contact_definitions_acf_decay",
        x_max_ns=2000.0,
    )


if __name__ == "__main__":
    main()
