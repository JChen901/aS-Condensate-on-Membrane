"""Replicate-resolved KS tests for persistent three-consecutive NAC contacts."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
POOLED = ROOT.parent
KS_ROOT = POOLED.parent
DATA = KS_ROOT.parent / "data"
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "hel_idp_persistent_replicate_ks_mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

FIG = ROOT / "figures"
TABLE = ROOT / "tables"
SYSTEMS = (
    ("G1_Wcluster_alhx0", "all", "IDP50", "#4C78A8"),
    ("G2_Wcluster_alhx700", "all", "Hel50", "#F58518"),
    ("G3_Onmem_plus", "all", "IDP50-Mem", "#54A24B"),
    ("G4_Onmem_wall", "IDP-IDP", "Hel10IDP40-Mem (IDP only)", "#B279A2"),
)
VARIANTS = ("strict_sym", "strict_anti", "banded_sym", "banded_anti")
VARIANT_LABELS = {
    "strict_sym": "strict par.",
    "strict_anti": "strict anti-par.",
    "banded_sym": "banded par.",
    "banded_anti": "banded anti-par.",
}
ARIAL_TTF_PATH = "/data/jychen/package/Arial/arial.ttf"


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


def hist(group: str, pair_type: str, rep: int, variant: str) -> np.ndarray:
    table = pd.read_csv(DATA / group / f"event_histograms_rep{rep}.csv")
    data = table[(table.pair_type == pair_type) & (table.variant == variant) & (table.contact == "three")]
    if len(data) == 0:
        return np.array([], dtype=np.int32)
    return np.repeat(data.lifetime_frames.to_numpy(dtype=np.int32), data["count"].to_numpy(dtype=np.int64))


def build_tests() -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    manifest = {}
    for group, _pair_type, _label, _color in SYSTEMS:
        for rep in (1, 2, 3):
            path = DATA / group / f"event_histograms_rep{rep}.csv"
            manifest[str(path)] = sha256(path)

    for variant in VARIANTS:
        for rep in (1, 2, 3):
            samples = {label: hist(group, pair_type, rep, variant) for group, pair_type, label, _color in SYSTEMS}
            for label_a, label_b in combinations(samples, 2):
                a, b = samples[label_a], samples[label_b]
                result = ks_2samp(a, b, alternative="two-sided", method="asymp")
                rows.append({
                    "replicate": rep,
                    "variant": variant,
                    "contact": "three",
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
    summary = (table.groupby(["variant", "contact", "system_a", "system_b"], as_index=False)
               .agg(mean_ks_D=("ks_D", "mean"), sd_ks_D=("ks_D", "std"),
                    min_q_value=("q_value_global_BH", "min"),
                    n_significant=("significant_q_0_05", "sum")))
    table.to_csv(TABLE / "stats_replicate_ks_three_consecutive_tests.csv", index=False)
    summary.to_csv(TABLE / "stats_replicate_ks_three_consecutive_D_summary.csv", index=False)
    return table, manifest


def save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_ks_d(table: pd.DataFrame) -> None:
    labels = [item[2] for item in SYSTEMS]
    tick = ["IDP50", "Hel50", "IDP50-Mem", "Hel10IDP40-Mem\n(IDP only)"]
    fig, axes = plt.subplots(2, 2, figsize=(mm_to_inch(150), mm_to_inch(135)), constrained_layout=True)
    vmax = max(.22, float(table.ks_D.max()))
    for ax, variant in zip(axes.flat, VARIANTS):
        matrix = np.full((4, 4), np.nan)
        sub = table[table.variant == variant]
        for _, rows in sub.groupby(["system_a", "system_b"], sort=False):
            a, b = rows.system_a.iloc[0], rows.system_b.iloc[0]
            i, j = labels.index(a), labels.index(b)
            matrix[max(i, j), min(i, j)] = rows.ks_D.mean()
        image = ax.imshow(matrix, vmin=0, vmax=vmax, cmap="viridis")
        for i in range(4):
            for j in range(4):
                if i <= j:
                    continue
                rows = sub[((sub.system_a == labels[i]) & (sub.system_b == labels[j])) |
                           ((sub.system_a == labels[j]) & (sub.system_b == labels[i]))]
                values = rows.sort_values("replicate").ks_D.to_numpy()
                value = float(values.mean())
                ax.text(j, i, "\n".join(f"{x:.3f}" for x in values), ha="center", va="center",
                        fontsize=8.5, color="white" if value > .12 else "black")
        ax.set_xticks(range(4), tick, rotation=35, ha="right")
        ax.set_yticks(range(4), tick)
        ax.set_title(VARIANT_LABELS[variant])
        ax.set_xlim(-.5, 3.5)
        ax.set_ylim(3.5, -.5)
    fig.colorbar(image, ax=axes, shrink=.83, label="Mean KS statistic D across three trajectories")
    fig.suptitle("Three-consecutive contacts: replicate-resolved KS D\nLower triangle: rep1, rep2, rep3")
    save(fig, "replicate_ks_D_three_consecutive")


def plot_replicate_ecdf() -> None:
    fig, axes = plt.subplots(4, 3, figsize=(mm_to_inch(180), mm_to_inch(180)), sharex=True, sharey=True)
    colors = {label: color for _group, _pair_type, label, color in SYSTEMS}
    for row, variant in enumerate(VARIANTS):
        for col, rep in enumerate((1, 2, 3)):
            ax = axes[row, col]
            for group, pair_type, label, _color in SYSTEMS:
                values = hist(group, pair_type, rep, variant)
                shown = values[values <= 30]
                x, n = np.unique(shown, return_counts=True)
                ax.step(np.r_[0, x], np.r_[0, np.cumsum(n) / len(values)], where="post",
                        color=colors[label], lw=1.6,
                        ls="--" if pair_type == "IDP-IDP" else "-", label=label)
            if row == 0:
                ax.set_title(f"rep{rep}")
            if col == 0:
                ax.set_ylabel(VARIANT_LABELS[variant] + "\nECDF")
            ax.set_xlim(0, 30)
            ax.set_ylim(0, 1.02)
            ax.tick_params(direction="in", top=True, right=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, .955))
    fig.supxlabel("Lifetime (ns)", y=.035)
    fig.suptitle("Persistent largest-cluster chains: three-consecutive ECDFs", y=.995)
    fig.subplots_adjust(left=.09, right=.995, bottom=.08, top=.88, hspace=.18, wspace=.08)
    save(fig, "replicate_ecdf_three_consecutive")


def main() -> None:
    configure_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    table, manifest = build_tests()
    plot_ks_d(table)
    plot_replicate_ecdf()
    (ROOT / "input_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "n_tests": len(table),
        "tests_per_definition": int(len(table) / len(VARIANTS)),
        "definitions": list(VARIANTS),
        "contact": "three",
        "n_significant_global_BH_q_0_05": int(table.significant_q_0_05.sum()),
        "input_files": len(manifest),
        "input_files_unchanged": all(sha256(Path(path)) == digest for path, digest in manifest.items()),
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
