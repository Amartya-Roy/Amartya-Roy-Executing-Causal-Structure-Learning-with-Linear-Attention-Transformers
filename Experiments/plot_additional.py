"""Column figures for the additional checks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path(__file__).resolve().parent / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "execution": "#1f4e79",
    "optimization": "#c47b2b",
    "statistical": "#5b8c5a",
    "float64": "#1f4e79",
    "float32": "#5b8c5a",
}


def _load():
    return json.loads((RES / "additional.json").read_text())


def _mean_se(values):
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, se


def error_decomposition(payload):
    rows = payload["decomposition"]
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.7))
    keys = (
        ("exec_finite", "execution"),
        ("opt", "optimization"),
        ("stat", "statistical"),
    )
    ps = (5, 10, 20)
    width = 0.24
    x = np.arange(len(ps))
    for ax, m, title in (
        (axes[0], 100, r"(1) $m=100$"),
        (axes[1], 1000, r"(2) $m=1000$"),
    ):
        for j, (field, label) in enumerate(keys):
            means, ses = [], []
            for p in ps:
                mean, se = _mean_se(
                    r[field]["frobenius"] for r in rows if r["p"] == p and r["m"] == m
                )
                means.append(max(mean, 1e-18))
                ses.append(se)
            ax.bar(
                x + (j - 1) * width, means, width, yerr=ses,
                color=COLORS[label], label=label, capsize=1.2,
                error_kw={"lw": 0.5, "capthick": 0.5, "ecolor": "0.2"},
            )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"$p={p}$" for p in ps])
        ax.set_ylabel(r"$\|W\|$ error", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=5.5,
               bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIG / "error_decomposition.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


def raw_input(payload):
    rows = payload["decomposition"]
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.35), constrained_layout=True)
    ps = (5, 10, 20)
    x = np.arange(len(ps))
    width = 0.34
    panels = (
        (axes[0], "sigma_head_float64", "sigma_head_float32", r"$\|\Sigma\|$ error", "(1) covariance head"),
        (axes[1], "raw_full_float64", "raw_full_float32", r"$\|W\|$ error", "(2) after the same updates"),
    )
    for ax, key64, key32, ylabel, title in panels:
        for shift, key, label, color in (
            (-width / 2, key64, "float64", COLORS["float64"]),
            (width / 2, key32, "float32", COLORS["float32"]),
        ):
            means, ses = [], []
            for p in ps:
                chunk = [r for r in rows if r["p"] == p and r["m"] == 1000]
                mean, se = _mean_se(r[key]["frobenius"] for r in chunk)
                means.append(mean if mean > 0 else np.nan)
                ses.append(se if mean > 0 else 0.0)
            ax.bar(
                x + shift, means, width, yerr=ses, color=color, label=label, capsize=1.2,
                error_kw={"lw": 0.5, "capthick": 0.5, "ecolor": "0.2"},
            )
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"$p={p}$" for p in ps])
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6.5)
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper left")
    path = FIG / "raw_input.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


def scaling(payload):
    rows = [r for r in payload["scaling"] if r["p"] == r["p_max"]]
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 4.15), constrained_layout=True)
    for name, color, marker in (("float64", COLORS["float64"], "s"), ("float32", COLORS["float32"], "o")):
        chunk = sorted([r for r in rows if r["dtype"] == name], key=lambda r: r["p_max"])
        xs = [r["p_max"] for r in chunk]
        axes[0].plot(xs, [max(r["W_residual_max"], 1e-18) for r in chunk], color=color, marker=marker, ms=3.5, lw=1.3, label=name)
        axes[2].plot(xs, [1e3 * r["seconds_median"] for r in chunk], color=color, marker=marker, ms=3.5, lw=1.3, label=name)
    chunk64 = sorted([r for r in rows if r["dtype"] == "float64"], key=lambda r: r["p_max"])
    axes[1].plot([r["p_max"] for r in chunk64], [r["layers"] for r in chunk64], color="#1f4e79", marker="o", ms=3.5, lw=1.3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"max $\|W\|$ residual", fontsize=7)
    axes[0].set_title("(1) one update at $p=p_{\\max}$", fontsize=8)
    axes[1].set_ylabel("layers", fontsize=7)
    axes[1].set_title(r"(2) depth $2s+8$", fontsize=8)
    axes[2].set_ylabel("milliseconds", fontsize=7)
    axes[2].set_xlabel(r"$p_{\max}$")
    axes[2].set_title("(3) time for one update", fontsize=8)
    axes[2].legend(frameon=False, fontsize=5.5, loc="upper left")
    for ax in axes:
        ax.set_xticks([16, 32, 64, 128])
        ax.tick_params(labelsize=6.5)
    path = FIG / "scaling.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


def sachs_heatmaps():
    pack = np.load(RES / "sachs_graphs.npz")
    # Show the baseline with the smaller structural Hamming distance.
    from experiments.metrics import shd

    truth = pack["truth"]
    choices = ("notears", "dagma", "golem")
    best = min(choices, key=lambda name: shd(truth, pack[name], 0.3))
    panels = (
        ("reference", truth),
        ("Poly-prox", pack["poly_prox"]),
        ("transformer", pack["transformer"]),
        (best.upper() if best != "notears" else "NOTEARS", pack[best]),
    )
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.45), constrained_layout=True)
    ys, xs = np.nonzero(truth)
    for ax, (title, W) in zip(axes.ravel(), panels):
        show = np.abs(W)
        vmax = max(float(show.max()), 1e-8)
        ax.imshow(show / vmax, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax.scatter(xs, ys, s=10, c="#e23b3b", edgecolors="none")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    path = FIG / "sachs_heatmaps.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path, "baseline", best)


def baseline_shd(payload):
    rows = [r for r in payload["baselines"] if r["shd"] is not None]
    order = (
        ("poly_prox", "Poly-prox"),
        ("poly_lbfgs", "Poly-LBFGS"),
        ("notears", "NOTEARS"),
        ("dagma", "DAGMA"),
        ("golem", "GOLEM"),
        ("sort", "Sort"),
        ("pc", "PC"),
        ("ges", "GES"),
    )
    colors = {
        "poly_prox": "#1f4e79",
        "poly_lbfgs": "#4c78a8",
        "notears": "#5b8c5a",
        "dagma": "#c47b2b",
        "golem": "#8c3a4b",
        "sort": "#d4a017",
        "pc": "#6b5b95",
        "ges": "#4c4c4c",
    }
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.15))
    ps = (5, 10, 20)
    x = np.arange(len(ps))
    width = 0.1
    for ax, scale, title in (
        (axes[0], "raw", "(1) raw"),
        (axes[1], "standardized", "(2) standardized"),
    ):
        for j, (key, label) in enumerate(order):
            means, ses = [], []
            for p in ps:
                chunk = [r["shd"] for r in rows if r["method"] == key and r["p"] == p and r["scale"] == scale]
                mean, se = _mean_se(chunk)
                means.append(mean)
                ses.append(se)
            ax.bar(
                x + (j - 3.5) * width, means, width, yerr=ses, color=colors[key], label=label,
                capsize=0.8, error_kw={"lw": 0.4, "capthick": 0.4, "ecolor": "0.2"},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"$p={p}$" for p in ps])
        ax.set_ylabel("mean SHD", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=6.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=5,
               bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = FIG / "baseline_shd.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


def external_inherit():
    """Recovery tracks Poly-prox; execution error is a separate axis."""
    rows = []
    for name in ("external.json", "external_alarm.json"):
        path = RES / name
        if path.exists():
            rows.extend(json.loads(path.read_text()).get("rows", []))
    if not rows:
        return
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 3.45), constrained_layout=True)
    colors = {
        "child": "#1f4e79",
        "alarm": "#c47b2b",
        "other": "#5b8c5a",
    }
    ax = axes[0, 0]
    for topology, color, label in (
        ("child", colors["child"], "CHILD"),
        ("alarm", colors["alarm"], "ALARM"),
    ):
        chunk = [r for r in rows if r["topology"] == topology]
        if not chunk:
            continue
        ax.scatter(
            [r["scores"]["poly_prox"]["shd"] for r in chunk],
            [r["scores"]["transformer"]["shd"] for r in chunk],
            s=16, c=color, label=label, zorder=3,
        )
    others = [r for r in rows if r["topology"] not in ("child", "alarm")]
    if others:
        ax.scatter(
            [r["scores"]["poly_prox"]["shd"] for r in others],
            [r["scores"]["transformer"]["shd"] for r in others],
            s=12, c=colors["other"], label="other topologies", zorder=2,
        )
    xmax = max((r["scores"]["poly_prox"]["shd"] for r in rows), default=1)
    ax.plot([0, xmax], [0, xmax], color="0.45", lw=0.8, zorder=1)
    ax.set_xlabel("Poly-prox SHD")
    ax.set_ylabel("transformer SHD")
    ax.set_title("(a) recovery follows the solver", fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.tick_params(labelsize=7)

    ax = axes[0, 1]
    order = [name for name in ("asia", "cancer", "earthquake", "survey", "sachs", "child", "alarm")
             if any(r["topology"] == name for r in rows)]
    ax.axhline(0.0, color="#1f4e79", lw=1.4)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel(r"$\|W_T-W_P\|_F$")
    ax.set_title("(b) execution error is 0", fontsize=8)
    ax.tick_params(labelsize=7)

    curve_methods = (
        ("poly_prox", "Poly-prox", "#1f4e79", "o", "-"),
        ("transformer", "transformer", "#4c78a8", "s", "--"),
        ("poly_lbfgs", "Poly-LBFGS", "#5b8c5a", "^", "-"),
        ("notears", "NOTEARS", "#c47b2b", "D", "-"),
    )
    for ax, topology, title in (
        (axes[1, 0], "child", "(c) CHILD"),
        (axes[1, 1], "alarm", "(d) ALARM"),
    ):
        chunk = [r for r in rows if r["topology"] == topology]
        ms = sorted({r["m"] for r in chunk})
        for key, label, color, marker, style in curve_methods:
            means, ses = [], []
            for m in ms:
                mean, se = _mean_se(r["scores"][key]["shd"] for r in chunk if r["m"] == m)
                means.append(mean)
                ses.append(se)
            if ms:
                ax.errorbar(ms, means, yerr=ses, color=color, marker=marker, ms=3.2, lw=1.1,
                            ls=style, label=label, capsize=1.5)
        ax.set_xscale("log")
        if ms:
            ax.set_xticks(ms)
            ax.set_xticklabels([str(m) for m in ms], fontsize=6)
        ax.set_xlabel("sample size")
        ax.set_ylabel("mean SHD")
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[1, 0].legend(frameon=False, fontsize=5.5, ncol=2, loc="lower left")
    path = FIG / "inherit.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path, "rows", len(rows))


def _inherit_calls(truth, poly, transformer):
    true = np.abs(truth) > 0
    left = np.abs(poly) > 0.3
    right = np.abs(transformer) > 0.3
    np.fill_diagonal(true, False)
    np.fill_diagonal(left, False)
    np.fill_diagonal(right, False)
    out = np.zeros(truth.shape, dtype=np.int8)
    out[true & left & right] = 1
    out[true & ~left & ~right] = 2
    out[~true & left & right] = 3
    out[left != right] = 4
    return out


def _inherit_heatmap_page(pack, names, labels, height, outfile):
    """One full-width block, short enough to share a page with text."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    call_cmap = ListedColormap(["#f2f2f2", "#1f7a3a", "#e23b3b", "#e0a100", "#111111"])
    fig = plt.figure(figsize=(6.75, height))
    gs = fig.add_gridspec(
        len(names), 6,
        width_ratios=[1, 0.045, 1, 1, 0.045, 1],
        left=0.125, right=0.90, top=0.90, bottom=0.145,
        wspace=0.05, hspace=0.12,
    )
    columns = (0, 2, 3, 5)
    weight_axes = []
    for r, (name, label) in enumerate(zip(names, labels)):
        truth = pack[f"{name}_truth"]
        poly = pack[f"{name}_poly"]
        transformer = pack[f"{name}_transformer"]
        vmax = max(float(np.max(np.abs(M))) for M in (truth, poly, transformer))
        vmax = max(vmax, 1e-8)
        panels = (
            (np.abs(truth) / vmax, "viridis", 0.0, 1.0),
            (np.abs(poly) / vmax, "viridis", 0.0, 1.0),
            (np.abs(transformer) / vmax, "viridis", 0.0, 1.0),
            (_inherit_calls(truth, poly, transformer), call_cmap, -0.5, 4.5),
        )
        row_axes = []
        for (show, cmap, vmin, vmax_show), c in zip(panels, columns):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(show, cmap=cmap, vmin=vmin, vmax=vmax_show, interpolation="nearest", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("0.45")
                spine.set_linewidth(0.4)
            row_axes.append(ax)
            if c != 5:
                weight_axes.append((ax, truth))
        row_axes[0].set_ylabel(f"{label}\nSHD {int(pack[f'{name}_shd'][0])}", fontsize=6.5)
        if r == 0:
            for ax, title in zip(row_axes, ("reference", "Poly-prox", "transformer", "both methods")):
                ax.set_title(title, fontsize=8, pad=2)

    fig.canvas.draw()
    for ax, truth in weight_axes:
        ii, jj = np.nonzero(np.abs(truth) > 0)
        bbox = ax.get_window_extent()
        cell = min(bbox.width, bbox.height) / truth.shape[0]
        diameter = 0.62 * cell / fig.dpi * 72
        ax.scatter(jj, ii, s=diameter ** 2, c="#e23b3b", edgecolors="none", linewidths=0, zorder=3)

    ref = fig.axes[0].get_position()
    prox = fig.axes[1].get_position()
    block = fig.axes[2].get_position()
    call = fig.axes[3].get_position()
    last = fig.axes[-1].get_position()
    for x in (0.5 * (ref.x1 + prox.x0), 0.5 * (block.x1 + call.x0)):
        fig.add_artist(plt.Line2D(
            [x, x], [last.y0, ref.y1],
            transform=fig.transFigure, color="black", linewidth=0.7, clip_on=False,
        ))
    cax = fig.add_axes([0.915, last.y0, 0.015, ref.y1 - last.y0])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 1.0), cmap="viridis"),
        cax=cax,
    )
    cax.tick_params(labelsize=6)
    cax.set_ylabel("edge strength", fontsize=7)
    fig.legend(
        handles=[
            Patch(facecolor="#1f7a3a", edgecolor="none", label="kept by both"),
            Patch(facecolor="#e23b3b", edgecolor="none", label="missed by both"),
            Patch(facecolor="#e0a100", edgecolor="none", label="extra edge, both"),
            Patch(facecolor="#111111", edgecolor="none", label="they disagree"),
        ],
        loc="lower center", ncol=4, frameon=False, fontsize=6.5,
        bbox_to_anchor=(0.51, 0.005),
    )
    fig.savefig(outfile)
    fig.savefig("/tmp/" + Path(outfile).stem + ".png", dpi=150)
    plt.close(fig)
    print(outfile)


def inherit_heatmaps():
    """Two half-page figures: small topologies, then SACHS, CHILD, and ALARM."""
    path = RES / "inherit_graphs.npz"
    if not path.exists():
        return
    pack = np.load(path)
    _inherit_heatmap_page(
        pack,
        ("asia", "cancer", "earthquake", "survey"),
        ("ASIA", "CANCER", "EARTHQUAKE", "SURVEY"),
        3.15,
        FIG / "inherit_heatmaps_small.pdf",
    )
    _inherit_heatmap_page(
        pack,
        ("sachs", "child", "alarm"),
        ("SACHS", "CHILD", "ALARM"),
        3.45,
        FIG / "inherit_heatmaps_large.pdf",
    )


def external_heatmaps():
    from experiments.metrics import shd

    pack = np.load(RES / "external_child_m10000.npz")
    truth = pack["truth"]
    draws = []
    for draw in range(10):
        key = f"poly_{draw}"
        if key not in pack.files:
            continue
        draws.append((shd(truth, pack[key], 0.3), draw))
    if not draws:
        return
    shds = np.array([s for s, _ in draws], dtype=float)
    median = float(np.median(shds))
    _, draw = min(draws, key=lambda item: (abs(item[0] - median), item[1]))
    panels = (
        (r"reference $W^\dagger$", truth),
        ("Poly-prox", pack[f"poly_{draw}"]),
        ("transformer", pack[f"tr_{draw}"]),
        ("NOTEARS", pack[f"nt_{draw}"]),
    )
    mats = [np.asarray(W, dtype=float) for _, W in panels]
    vmax = max(float(np.max(np.abs(W))) for W in mats)
    vmax = max(vmax, 1e-8)
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.55), constrained_layout=True)
    ys, xs = np.nonzero(np.abs(truth) > 0)
    image = None
    for ax, (title, W) in zip(axes.ravel(), panels):
        image = ax.imshow(W, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.scatter(xs, ys, s=14, facecolors="none", edgecolors="0.1", linewidths=0.5)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, label="edge weight")
    path = FIG / "external_heatmaps.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path, "draw", draw, "median shd", median)


if __name__ == "__main__":
    payload = _load()
    if "decomposition" in payload:
        error_decomposition(payload)
        raw_input(payload)
    if "scaling" in payload:
        scaling(payload)
    if (RES / "sachs_graphs.npz").exists():
        sachs_heatmaps()
    if "baselines" in payload:
        baseline_shd(payload)
    if (RES / "external.json").exists():
        external_inherit()
    if (RES / "inherit_graphs.npz").exists():
        inherit_heatmaps()
    if (RES / "external_child_m10000.npz").exists():
        external_heatmaps()
