import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARIAL_TTF_PATH = "/data/jychen/package/Arial/arial.ttf"
USE_ARIAL_TTF = True
_PUB_FONT_NAME = None

FIG_SIZE = {
    "single": 85,
    "double": 170,
}

FIG_PRESETS = {
    "single_small": ("single", 60),
    "single_short": ("single", 70),
    "single_medium": ("single", 95),
    "double_short": ("double", 70),
    "double_medium": ("double", 100),
    "double_large": ("double", 155),
}

DENSITY_FREE_ENERGY_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")

CONTACT_MODE_SHORT_LABELS = {
    "any_nac_to_nac_bb": "Any BB-BB",
    "three_symmetric_nac_bb": r"$\geq$3 band sym.",
    "symmetric_nac_bb": r"$\geq$1 band sym.",
    "triplet_symmetric_nac_bb": "3 band sym consec.",
}

CONTACT_MODE_PANEL_LABELS = {
    "any_nac_to_nac_bb": "Any BB",
    "three_symmetric_nac_bb": r"$\geq$3 band sym.",
    "symmetric_nac_bb": r"$\geq$1 band sym.",
    "triplet_symmetric_nac_bb": "3 band sym consec.",
}

CONTACT_MODE_TITLES = {
    "any_nac_to_nac_bb": "Any BB-BB contact",
    "three_symmetric_nac_bb": "Any 3 banded symmetric BB contacts",
    "symmetric_nac_bb": "Any 1 banded symmetric BB contact",
    "triplet_symmetric_nac_bb": "3 consecutive banded symmetric BB contacts",
}

PLOT_CONTEXT_LABELS = {
    "nac_contact_survival": "NAC contact survival",
    "nac_symmetric_contact_survival": "NAC symmetric contact survival",
    "nter_contact_survival": "N-ter contact survival",
    "cter_contact_survival": "C-ter contact survival",
    "nac_contact_acf": "NAC contact ACF",
    "nac_symmetric_contact_acf": "NAC symmetric contact ACF",
    "nter_contact_acf": "N-ter contact ACF",
    "cter_contact_acf": "C-ter contact ACF",
}

THRESHOLD_SPECS = (
    {"name": "t_1e", "label": "1/e", "threshold": 1.0 / np.e, "summary_col": "t_1e_ps"},
    {"name": "t_01", "label": "0.1", "threshold": 0.1, "summary_col": "t_01_ps"},
)


def load_pub_font():
    global _PUB_FONT_NAME

    if _PUB_FONT_NAME is not None:
        return _PUB_FONT_NAME

    if USE_ARIAL_TTF:
        if not os.path.exists(ARIAL_TTF_PATH):
            raise FileNotFoundError(f"Arial font file not found: {ARIAL_TTF_PATH}")
        fm.fontManager.addfont(ARIAL_TTF_PATH)
        _PUB_FONT_NAME = fm.FontProperties(fname=ARIAL_TTF_PATH).get_name()
    else:
        _PUB_FONT_NAME = "Arial"

    print(f"Using font: {_PUB_FONT_NAME}")
    return _PUB_FONT_NAME


def mm_to_inch(mm):
    return mm / 25.4


def set_pub_style():
    pub_font = load_pub_font()
    base = {
        "font.family": pub_font,
        "font.sans-serif": [pub_font, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
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
        "axes.unicode_minus": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.dpi": 600,
        "figure.dpi": 120,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
    mpl.rcParams.update(base)


def new_figure(mode="single", height_mm=65):
    set_pub_style()
    width_mm = FIG_SIZE[mode]
    return plt.subplots(
        1,
        1,
        figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)),
    )


def new_subplots(nrows, ncols, preset="single_medium", **kwargs):
    set_pub_style()
    mode, height_mm = FIG_PRESETS[preset]
    width_mm = FIG_SIZE[mode]
    return plt.subplots(
        nrows,
        ncols,
        figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)),
        **kwargs,
    )


def savefig_pub(fig, filename, dpi=600, tight=False):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(filename, dpi=dpi)


def style_axis_like_msd(ax, hide_minor=True):
    ax.grid(False)
    ax.tick_params(top=False, right=False)
    if hide_minor:
        ax.minorticks_off()


def default_group_labels(groups):
    labels = {
        "G1_Wcluster_alhx0": r"$\mathrm{IDP}_{50}$",
        "G2_Wcluster_alhx700": r"$\mathrm{Hel}_{50}$",
        "G3_Onmem_plus": r"$\mathrm{IDP}_{50}$-Mem",
        "G4_Onmem_wall": r"$\mathrm{Hel}_{10}\mathrm{IDP}_{40}$-Mem",
    }
    return {group["name"]: labels.get(group["name"], group.get("label", group["name"])) for group in groups}


def default_group_colors(groups):
    return {
        group["name"]: DENSITY_FREE_ENERGY_COLORS[idx]
        for idx, group in enumerate(groups)
    }


def plot_context_label(file_prefix):
    return PLOT_CONTEXT_LABELS.get(file_prefix, file_prefix.replace("_", " "))


def threshold_axis_label(file_prefix, metric_spec):
    if "survival" in file_prefix:
        return rf"$\mathrm{{T}}_{{\mathrm{{S}}}}({metric_spec['label']})$ (ns)"
    return rf"$\mathrm{{T}}_{{\mathrm{{ACF}}}}({metric_spec['label']})$ (ns)"


def crossing_time_ps(time_ps, values, threshold):
    values = np.asarray(values, dtype=float)
    time_ps = np.asarray(time_ps, dtype=float)
    valid = np.isfinite(time_ps) & np.isfinite(values)
    values = values[valid]
    time_ps = time_ps[valid]
    if len(values) == 0:
        return np.nan

    below = np.where(values <= threshold)[0]
    if len(below) == 0:
        return np.nan

    idx = below[0]
    if idx == 0:
        return float(time_ps[0])

    t0, t1 = time_ps[idx - 1], time_ps[idx]
    y0, y1 = values[idx - 1], values[idx]
    if y1 == y0:
        return float(t1)
    return float(t0 + (threshold - y0) * (t1 - t0) / (y1 - y0))


def symmetric_error(*errors):
    finite_errors = [float(error) for error in errors if np.isfinite(error)]
    if not finite_errors:
        return np.nan
    return max(0.0, max(finite_errors))


def threshold_time_with_error(df_group, mean_col, std_col, threshold, summary_time_ps=np.nan):
    df_group = df_group.sort_values("time_ps")
    time_ps = df_group["time_ps"].to_numpy(dtype=float)
    mean = df_group[mean_col].to_numpy(dtype=float)
    std = df_group[std_col].to_numpy(dtype=float)

    time_mid = float(summary_time_ps) if np.isfinite(summary_time_ps) else crossing_time_ps(time_ps, mean, threshold)
    if not np.isfinite(time_mid):
        return np.nan, np.nan, np.nan, np.nan, False

    lower_cross = crossing_time_ps(time_ps, mean - std, threshold)
    upper_cross = crossing_time_ps(time_ps, mean + std, threshold)
    err_low = max(0.0, time_mid - lower_cross) if np.isfinite(lower_cross) else np.nan
    err_high = max(0.0, upper_cross - time_mid) if np.isfinite(upper_cross) else np.nan
    err_sym = symmetric_error(err_low, err_high)
    return time_mid, err_low, err_high, err_sym, True


def metric_rows_from_results(results, mean_col, std_col, metric_spec, contact_modes, groups, group_labels):
    rows = []
    for contact_mode in contact_modes:
        result = results.get(contact_mode)
        if result is None:
            continue
        df_all = result.get("data", pd.DataFrame())
        summary = result.get("summary", pd.DataFrame())
        if df_all.empty:
            continue

        for group in groups:
            group_name = group["name"]
            df_group = df_all[df_all["group"] == group_name]
            summary_time_ps = np.nan
            if not summary.empty and metric_spec["summary_col"] in summary:
                summary_match = summary[
                    (summary["group"] == group_name)
                    & (summary["contact_mode"] == contact_mode)
                ]
                if not summary_match.empty:
                    summary_time_ps = float(summary_match.iloc[0][metric_spec["summary_col"]])

            if df_group.empty:
                reached = False
                time_ps = err_low_ps = err_high_ps = err_sym_ps = np.nan
            else:
                time_ps, err_low_ps, err_high_ps, err_sym_ps, reached = threshold_time_with_error(
                    df_group,
                    mean_col,
                    std_col,
                    metric_spec["threshold"],
                    summary_time_ps=summary_time_ps,
                )

            rows.append(
                {
                    "group": group_name,
                    "group_label": group_labels.get(group_name, group_name),
                    "contact_mode": contact_mode,
                    "contact_label": CONTACT_MODE_TITLES.get(contact_mode, contact_mode),
                    "threshold": metric_spec["label"],
                    "threshold_reached": bool(reached),
                    "time_ps": time_ps,
                    "time_ns": time_ps / 1e3 if np.isfinite(time_ps) else np.nan,
                    "err_low_ps": err_low_ps,
                    "err_low_ns": err_low_ps / 1e3 if np.isfinite(err_low_ps) else np.nan,
                    "err_high_ps": err_high_ps,
                    "err_high_ns": err_high_ps / 1e3 if np.isfinite(err_high_ps) else np.nan,
                    "err_sym_ps": err_sym_ps,
                    "err_sym_ns": err_sym_ps / 1e3 if np.isfinite(err_sym_ps) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _read_mode_tables(output_dir, data_suffix, contact_modes):
    output_dir = Path(output_dir)
    data_by_mode = {}
    summary_by_mode = {}

    for path in sorted(output_dir.glob(f"*/*{data_suffix}")):
        df = pd.read_csv(path)
        if df.empty or "contact_mode" not in df:
            continue
        mode = str(df["contact_mode"].dropna().iloc[0])
        data_by_mode[mode] = df

    for path in sorted(output_dir.glob("*/*_summary.csv")):
        if path.name.endswith("_trajectory_summary.csv"):
            continue
        df = pd.read_csv(path)
        if df.empty or "contact_mode" not in df:
            continue
        mode = str(df["contact_mode"].dropna().iloc[0])
        summary_by_mode[mode] = df

    results = {
        mode: {
            "data": data_by_mode.get(mode, pd.DataFrame()),
            "summary": summary_by_mode.get(mode, pd.DataFrame()),
        }
        for mode in contact_modes
        if mode in data_by_mode
    }
    complete = all(mode in data_by_mode for mode in contact_modes)
    combined_summary = (
        pd.concat([summary_by_mode[mode] for mode in contact_modes if mode in summary_by_mode], ignore_index=True)
        if summary_by_mode
        else pd.DataFrame()
    )
    return results, combined_summary, complete


def load_cached_survival(output_dir, contact_modes):
    return _read_mode_tables(output_dir, "_survival_data.csv", contact_modes)


def load_cached_acf(output_dir, contact_modes):
    return _read_mode_tables(output_dir, "_acf_data.csv", contact_modes)


def draw_reference_line(ax, y_value, text, color, linewidth=1.0):
    ax.axhline(y_value, color=color, ls="--", lw=linewidth, alpha=0.85)
    ax.text(
        0.98,
        y_value,
        text,
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        color=color,
        fontsize=mpl.rcParams["legend.fontsize"],
    )


def _align_edge_xticklabels(ax):
    labels = ax.get_xticklabels()
    if labels:
        labels[0].set_ha("left")
        labels[-1].set_ha("right")


def plot_decay_final_summary(
    results,
    mean_col,
    std_col,
    output_dir,
    file_prefix,
    ylabel,
    xlabel,
    xlim_ns,
    contact_modes,
    groups,
    group_labels,
    group_colors,
    y_min=-0.05,
    preset="double_large",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = new_subplots(2, 2, preset=preset, sharex=True, sharey=True)
    axes = np.ravel(axes)
    legend_handles = {}

    for ax, contact_mode in zip(axes, contact_modes):
        result = results.get(contact_mode)
        if result is None or result["data"].empty:
            ax.set_visible(False)
            continue

        ax.text(
            0.04,
            0.92,
            CONTACT_MODE_PANEL_LABELS.get(contact_mode, contact_mode),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=mpl.rcParams["legend.fontsize"],
        )

        for group in groups:
            group_name = group["name"]
            d = result["data"][result["data"]["group"] == group_name].sort_values("time_ps")
            if d.empty:
                continue

            t_ns = d["time_ps"].to_numpy(dtype=float) / 1e3
            mean = d[mean_col].to_numpy(dtype=float)
            std = d[std_col].to_numpy(dtype=float)
            color = group_colors.get(group_name, group.get("color", "#1f77b4"))
            label = group_labels.get(group_name, group_name)

            line, = ax.plot(t_ns, mean, color=color, lw=1.8, label=label)
            ax.fill_between(t_ns, mean - std, mean + std, color=color, alpha=0.14, lw=0)
            legend_handles.setdefault(label, line)

        draw_reference_line(ax, 1.0 / np.e, "1/e", "#2ca02c")
        draw_reference_line(ax, 0.1, "0.1", "#9467bd")
        ax.set_xlim(0, xlim_ns)
        ax.set_xticks([0, xlim_ns / 2, xlim_ns])
        _align_edge_xticklabels(ax)
        ax.set_ylim(y_min, 1.35)
        style_axis_like_msd(ax, hide_minor=True)

    for ax in axes[-2:]:
        ax.set_xlabel(xlabel)
    fig.text(0.035, 0.535, ylabel, rotation="vertical", va="center", ha="center", fontsize=12)

    visible_axes = [ax for ax in axes if ax.get_visible()]
    legend_ax = visible_axes[0] if visible_axes else axes[0]
    if legend_handles:
        legend_ax.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            ncol=1,
            borderaxespad=0.0,
            frameon=True,
            framealpha=0.75,
            facecolor="white",
            edgecolor="none",
            fontsize=mpl.rcParams["legend.fontsize"],
            handlelength=mpl.rcParams["legend.handlelength"],
        )

    out_png = output_dir / f"{file_prefix}_final_summary.png"
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.96, wspace=0.18, hspace=0.24)
    savefig_pub(fig, out_png)
    plt.show()
    print(f"Figure context: {plot_context_label(file_prefix)}; x-axis = {xlabel}; y-axis = {ylabel}")
    print(f"Saved final summary plot: {out_png}")
    return out_png


def plot_threshold_metric_summary(
    results,
    mean_col,
    std_col,
    output_dir,
    file_prefix,
    metric_spec,
    contact_modes,
    groups,
    group_labels,
    group_colors,
    preset="single_medium",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_df = metric_rows_from_results(
        results,
        mean_col,
        std_col,
        metric_spec,
        contact_modes,
        groups,
        group_labels,
    )
    csv_path = output_dir / f"{file_prefix}_{metric_spec['name']}_summary.csv"
    metric_df.to_csv(csv_path, index=False)

    fig, ax = new_subplots(1, 1, preset=preset)
    x_base = np.arange(len(contact_modes), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(groups))
    y_upper_values = []

    for offset, group in zip(offsets, groups):
        group_name = group["name"]
        group_df = metric_df[(metric_df["group"] == group_name) & (metric_df["threshold_reached"])]
        if group_df.empty:
            continue

        xs = []
        ys = []
        yerr_sym = []
        for mode_idx, contact_mode in enumerate(contact_modes):
            row = group_df[group_df["contact_mode"] == contact_mode]
            if row.empty:
                continue
            row = row.iloc[0]
            if not np.isfinite(row["time_ns"]):
                continue
            err_sym_ns = row["err_sym_ns"] if np.isfinite(row["err_sym_ns"]) else 0.0
            xs.append(x_base[mode_idx] + offset)
            ys.append(row["time_ns"])
            yerr_sym.append(err_sym_ns)
            y_upper_values.append(row["time_ns"] + err_sym_ns)

        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=yerr_sym,
                fmt="o",
                ms=4.5,
                capsize=3,
                lw=1.2,
                color=group_colors.get(group_name, group.get("color", "#1f77b4")),
                label=group_labels.get(group_name, group_name),
            )

    y_axis_label = threshold_axis_label(file_prefix, metric_spec)
    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [CONTACT_MODE_SHORT_LABELS.get(mode, mode) for mode in contact_modes],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel(y_axis_label)
    style_axis_like_msd(ax, hide_minor=True)
    ax.margins(x=0.08)

    if y_upper_values:
        y_top = max(y_upper_values)
        y_bottom = -0.06 * y_top if y_top > 0 else -0.05
        y_top_plot = y_top * 1.55 if y_top > 0 else 1.0
        ax.set_ylim(y_bottom, y_top_plot)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=1,
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.75,
        facecolor="white",
        edgecolor="none",
        fontsize=mpl.rcParams["legend.fontsize"],
        handlelength=mpl.rcParams["legend.handlelength"],
    )

    out_png = output_dir / f"{file_prefix}_{metric_spec['name']}_summary.png"
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.30, top=0.96)
    savefig_pub(fig, out_png)
    plt.show()
    print(
        f"Figure context: {plot_context_label(file_prefix)}; "
        f"threshold = {metric_spec['label']}; y-axis = {y_axis_label}"
    )
    print(f"Saved metric table: {csv_path}")
    print(f"Saved metric plot: {out_png}")
    if "threshold_reached" in metric_df:
        print(f"Reached threshold: {int(metric_df['threshold_reached'].sum())}/{len(metric_df)}")
    return metric_df, out_png


def plot_threshold_set(results, mean_col, std_col, output_dir, file_prefix, contact_modes, groups, group_labels, group_colors):
    metric_tables = {}
    for metric_spec in THRESHOLD_SPECS:
        metric_df, _ = plot_threshold_metric_summary(
            results,
            mean_col,
            std_col,
            output_dir,
            file_prefix,
            metric_spec,
            contact_modes,
            groups,
            group_labels,
            group_colors,
        )
        metric_tables[metric_spec["name"]] = metric_df
    return metric_tables
