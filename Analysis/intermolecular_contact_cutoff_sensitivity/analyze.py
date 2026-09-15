"""Frame-wise inter-chain contact-cutoff sensitivity (0.6/0.7/0.8/0.9/1.0 nm).

Reuses the BB selection, consecutive 140-bead chain IDs, PBC distance search,
and largest connected-component convention used in Analysis/BB_contact_shell.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/contact_cutoff_sensitivity_mpl")
from pathlib import Path
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import self_capped_distance, calc_bonds
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "data"
OUT.mkdir(exist_ok=True)
CUTOFFS = {0.6: "6A", 0.7: "7A", 0.8: "8A", 0.9: "9A", 1.0: "10A"}
CUTOFF_VALUES = tuple(CUTOFFS)
CUTOFF_A = tuple(int(c * 10) for c in CUTOFF_VALUES)
REFERENCE_CUTOFF_A = 8
RUNS = [f"alhx_{s}{r}" for s in (0, 700) for r in ("", "_re1", "_re2")]
COLORS = {"0": "#1f77b4", "700": "#d62728"}
LABELS = {"0": r"$\mathrm{IDP}_{50}$", "700": r"$\mathrm{Hel}_{50}$"}
MARKERS = {"0": "o", "700": "s"}

# Publication style copied from Net_Dist.ipynb, cell 0.
# 统一论文图的风格和尺寸, 方便后续拼接
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm   # 修改 1：新增，用于手动加载 Arial.ttf


# =========================
# 修改 2：只需要改这里
# =========================
ARIAL_TTF_PATH = "/data/jychen/package/Arial/arial.ttf"   # 改成你的 arial.ttf 实际路径
USE_ARIAL_TTF = True                # True = 从 ttf 文件加载 Arial；False = 直接用系统字体名


_PUB_FONT_NAME = None


def load_pub_font():
    """
    手动加载 Arial.ttf，并返回字体在 matplotlib 中识别到的真实名称。
    这样可以避免系统装了 Arial 但 matplotlib 仍然找不到的问题。
    """
    global _PUB_FONT_NAME

    if _PUB_FONT_NAME is not None:
        return _PUB_FONT_NAME

    if USE_ARIAL_TTF:
        if not os.path.exists(ARIAL_TTF_PATH):
            raise FileNotFoundError(f"找不到 Arial 字体文件: {ARIAL_TTF_PATH}")

        fm.fontManager.addfont(ARIAL_TTF_PATH)
        _PUB_FONT_NAME = fm.FontProperties(fname=ARIAL_TTF_PATH).get_name()
    else:
        _PUB_FONT_NAME = "Arial"

    print(f"Using font: {_PUB_FONT_NAME}")
    return _PUB_FONT_NAME


def mm_to_inch(mm):
    return mm / 25.4


FIG_SIZE = {
    "single": 90,
    "double": 180,
}


FIG_PRESETS = {
    "single_small":  ("single", 60),
    "single_medium": ("single", 75),
    "double_short":  ("double", 70),
    "double_medium": ("double", 100),
    "double_large":  ("double", 140),
}


_SHARED_FONT_SIZES = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
}


def set_pub_style():
    pub_font = load_pub_font()   # 修改 3：每次设置风格前，先确保 Arial.ttf 已经加载

    base = {
        # 修改 4：不再直接写死 "Arial"，而是使用 ttf 文件真实识别到的字体名
        "font.family": pub_font,
        "font.sans-serif": [pub_font, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],

        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # 稍粗一些的论文图风格
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
    }

    base.update(_SHARED_FONT_SIZES)
    mpl.rcParams.update(base)


def new_figure(mode="single", height_mm=65):
    set_pub_style()
    width_mm = FIG_SIZE[mode]
    fig, ax = plt.subplots(
        figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm))
    )
    return fig, ax


def new_subplots(nrows, ncols, preset="double_medium", **kwargs):
    set_pub_style()
    mode, height_mm = FIG_PRESETS[preset]
    width_mm = FIG_SIZE[mode]

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)),
        **kwargs
    )
    return fig, axes


def savefig_pub(fig, filename, dpi=600, tight=False):
    if tight:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(filename, dpi=dpi)




def largest_component(adj, n):
    if not adj:
        return np.array([0], dtype=int)
    rows = np.array([x[0] for x in adj], dtype=int)
    cols = np.array([x[1] for x in adj], dtype=int)
    mat = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    ncomp, labels = connected_components(mat, directed=False)
    sizes = np.bincount(labels, minlength=ncomp)
    return np.flatnonzero(labels == np.argmax(sizes))

def frame_result(pos, box, chain_id, nchain):
    # MDAnalysis coordinates are Å; CUTOFFS are stored in nm.
    pairs, _ = self_capped_distance(pos, max(CUTOFFS) * 10.0, box=box, method="nsgrid")
    if len(pairs) == 0:
        return {c: (np.array([0]), np.zeros(1, int)) for c in CUTOFFS}
    keep = chain_id[pairs[:, 0]] != chain_id[pairs[:, 1]]
    pairs = pairs[keep]
    d = calc_bonds(pos[pairs[:, 0]], pos[pairs[:, 1]], box=box) / 10.0
    out = {}
    for cutoff in CUTOFF_VALUES:
        q = pairs[d < cutoff]
        if len(q):
            e = np.unique(np.sort(np.column_stack((chain_id[q[:, 0]], chain_id[q[:, 1]])), axis=1), axis=0)
            largest = largest_component([(int(a), int(b)) for a, b in e], nchain)
            deg = np.zeros(nchain, dtype=int)
            for a, b in e:
                if a in largest and b in largest:
                    deg[a] += 1; deg[b] += 1
        else:
            largest = np.array([0], dtype=int); deg = np.zeros(nchain, dtype=int)
        out[cutoff] = (largest, deg[largest])
    return out

def compute(run):
    npz = OUT / f"{run}.npz"
    if npz.exists():
        with np.load(npz) as old:
            if "cutoffs_nm" in old and tuple(old["cutoffs_nm"]) == CUTOFF_VALUES:
                return
    folder = ROOT / "W_cluster" / run / "TR_Mdvwhole"
    u = mda.Universe(str(folder / "pbc.tpr"), str(folder / "clus_centered.xtc"))
    bb = u.select_atoms("name BB")
    assert len(bb) == 7000
    nchain, per = 50, 140
    chain_id = np.arange(len(bb)) // per
    assert np.all(bb.resnames.reshape(nchain, per) == bb.resnames[:per])
    # Existing BB analysis uses frames 10000..20000 (last 10 us), every 20 ns.
    frames = np.arange(10000, 20001, 20)
    ncut = len(CUTOFF_VALUES)
    n = len(frames); times = np.empty(n); sizes = np.zeros((n, ncut), int)
    degrees = np.zeros((n, ncut, nchain), int); members = np.zeros((n, ncut, nchain), bool)
    for k, fi in enumerate(frames):
        ts = u.trajectory[fi]; times[k] = ts.time
        r = frame_result(bb.positions, ts.dimensions, chain_id, nchain)
        for j, cutoff in enumerate(CUTOFF_VALUES):
            largest, selected_degree = r[cutoff]
            members[k, j] = np.isin(np.arange(nchain), largest)
            sizes[k, j] = len(largest); degrees[k, j][largest] = selected_degree
    np.savez_compressed(npz, frames=frames, times_ps=times, sizes=sizes, degrees=degrees, members=members, cutoffs_nm=np.array(CUTOFF_VALUES))

def summarize():
    rows, js, ts_rows, nd_rows = [], [], [], []
    for run in RUNS:
        z = np.load(OUT / f"{run}.npz"); system = run.split("_")[1]; rep = run.split("_", 2)[2] if "_" in run[run.index("_")+1:] else "1"
        if run.endswith("_re1"): rep = "2"
        elif run.endswith("_re2"): rep = "3"
        else: rep = "1"
        ref_idx = CUTOFF_A.index(REFERENCE_CUTOFF_A)
        for j, cutoff in enumerate(CUTOFF_VALUES):
            x = z["sizes"][:, j].astype(float); deg = z["degrees"][:, j];
            rows.append(dict(system=system, replicate=rep, cutoff_A=int(cutoff*10), mean_Nmax=x.mean(), sd_Nmax=x.std(ddof=1), median_Nmax=np.median(x), mean_neighbor=deg[deg>0].mean() if np.any(deg>0) else 0, sd_neighbor=deg[deg>0].std(ddof=1) if np.sum(deg>0)>1 else 0))
            for t, sz in zip(z["times_ps"], x): ts_rows.append(dict(system=system, replicate=rep, time_us=t/1e6, cutoff_A=int(cutoff*10), Nmax=int(sz)))
            for t, vals in zip(z["times_ps"], deg):
                for chain, degree in enumerate(vals):
                    if degree > 0: nd_rows.append(dict(system=system, replicate=rep, time_us=t/1e6, cutoff_A=int(cutoff*10), chain_id=chain, neighbor_number=int(degree)))
        for j, cutoff_a in enumerate(CUTOFF_A):
            if cutoff_a == REFERENCE_CUTOFF_A:
                continue
            label = f"{cutoff_a}_vs_{REFERENCE_CUTOFF_A}"
            a = z["members"][:, j]; b = z["members"][:, ref_idx]; J = (a & b).sum(1) / np.maximum((a | b).sum(1), 1)
            js.append(dict(system=system, replicate=rep, comparison=label, mean_J=J.mean(), sd_J=J.std(ddof=1), median_J=np.median(J), p05_J=np.percentile(J,5), p95_J=np.percentile(J,95)))
    summary = pd.DataFrame(rows); summary.to_csv(HERE/"summary.csv", index=False); pd.DataFrame(js).to_csv(HERE/"jaccard_summary.csv", index=False); pd.DataFrame(ts_rows).to_csv(HERE/"Nmax_time_series.csv", index=False); pd.DataFrame(nd_rows).to_csv(HERE/"neighbor_number_distribution.csv", index=False)
    plot(summary, pd.DataFrame(nd_rows))
    for system in sorted(summary.system.unique()):
        s = summary[summary.system == system].groupby("cutoff_A").mean(numeric_only=True); j = pd.DataFrame(js)[pd.DataFrame(js).system == system]; a=s.loc[REFERENCE_CUTOFF_A,"mean_Nmax"]
        changes = ", ".join([f"8->{c} {(s.loc[c,'mean_Nmax']/a-1)*100:+.2f}%" for c in CUTOFF_A if c != REFERENCE_CUTOFF_A])
        overlaps = ", ".join([f"J {c}vs8={j[j.comparison==f'{c}_vs_8'].mean_J.mean():.3f}" for c in CUTOFF_A if c != REFERENCE_CUTOFF_A])
        print(f"System alhx_{system}: Nmax {changes}; {overlaps}")
    for metric in ("mean_Nmax", "mean_neighbor"):
        orders = [tuple(summary[summary.cutoff_A==c].groupby("system")[metric].mean().sort_values(ascending=False).index.unique()) for c in CUTOFF_A]
        print(f"{metric} ranking: {orders}; unchanged={len(set(orders))==1}")

def plot(summary, neighbor_df):
    """Combine the largest-cluster trend and the notebook's five degree distributions.

    Trend error bars are SD across replicate means; distributions pool the
    sampled chains and frames, matching Net_Dist.ipynb's sensitivity cell.
    Jaccard statistics remain in the CSV output but are not plotted.
    """
    from matplotlib.ticker import MaxNLocator

    summary = summary.copy()
    neighbor_df = neighbor_df.copy()
    summary["system"] = summary["system"].astype(str)
    neighbor_df["system"] = neighbor_df["system"].astype(str)
    set_pub_style()
    fig = plt.figure(figsize=(mm_to_inch(180), mm_to_inch(130)), layout="constrained")
    fig.get_layout_engine().set(hspace=0.01, h_pad=0.025)
    top, bottom = fig.subfigures(2, 1, height_ratios=[0.96, 1], hspace=0.01)
    # Center the five-point trend in a narrower panel above the full-width distributions.
    top_grid = top.add_gridspec(1, 3, width_ratios=[1, 3, 1])
    cluster_ax = top.add_subplot(top_grid[0, 1])
    distributions = bottom.subplots(1, 5, sharex=True, sharey=True)
    bottom.supxlabel("Number of neighbors, n", fontsize=12)
    bottom.supylabel("Probability", fontsize=12)
    bottom.suptitle("B", x=0.015, ha="left", fontsize=26)

    for system, g in summary.groupby("system"):
        grouped = g.groupby("cutoff_A")
        means = grouped.mean(numeric_only=True)
        deviations = grouped["mean_Nmax"].std(ddof=1)
        cluster_ax.plot(g.cutoff_A / 10, g.mean_Nmax, linestyle="none", marker=MARKERS[system],
                        markersize=3.2, color=COLORS[system], alpha=0.45)
        cluster_ax.errorbar(means.index / 10, means.mean_Nmax, yerr=deviations,
                            color=COLORS[system], marker=MARKERS[system], markersize=4,
                            label=LABELS[system], capsize=3)
    cluster_ax.axvline(REFERENCE_CUTOFF_A / 10, color="0.65", linestyle="--", linewidth=1.2)
    cluster_ax.set(xlabel="Cutoff (nm)", ylabel="Mean largest cluster size",
                   xticks=CUTOFF_VALUES, yticks=[48, 49, 50], ylim=(47.9, 50.1))

    x_min = int(neighbor_df.neighbor_number.min())
    x_max = int(neighbor_df.neighbor_number.max())
    x = np.arange(x_min, x_max + 1)
    summary_rows = []
    for ax, cutoff in zip(distributions, CUTOFF_A):
        subset = neighbor_df[neighbor_df.cutoff_A == cutoff]
        for system in ("0", "700"):
            vals = subset.loc[subset.system == system, "neighbor_number"].to_numpy(dtype=int)
            if not len(vals):
                raise ValueError(f"No neighbor samples for system {system}, cutoff {cutoff / 10:.1f} nm")
            counts = np.bincount(vals - x_min, minlength=len(x)).astype(float)
            mean_n = float(vals.mean())
            summary_rows.append(dict(system=system, cutoff_A=cutoff,
                                     mean_n=mean_n, std_n=float(vals.std(ddof=1))))
            ax.plot(x, counts / counts.sum(), color=COLORS[system],
                    marker=MARKERS[system], markersize=3.2, linewidth=1.5,
                    label=LABELS[system])
            ax.axvline(mean_n, color=COLORS[system], linestyle="--", alpha=0.75,
                       linewidth=1.2)
        ax.set_title(f"{cutoff / 10:.1f} nm")
        ax.set_xticks([15, 30])
        ax.set_ylim(0.0, 0.2)
        ax.set_yticks([0.1, 0.2])
        ax.margins(x=0.02)
    for ax in distributions[1:]:
        ax.tick_params(labelleft=False)
    top.suptitle("A", x=0.015, ha="left", fontsize=26)
    cluster_ax.legend(loc="lower left")
    pd.DataFrame(summary_rows).to_csv(HERE / "neighbor_number_cutoff_summary.csv", index=False)
    for extension in ("png", "pdf"):
        savefig_pub(fig, HERE / f"intermolecular_contact_cutoff_sensitivity.{extension}")
    plt.close(fig)

if __name__ == "__main__":
    for run in RUNS: compute(run)
    summarize()
