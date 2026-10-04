"""Figures for the evaluation section."""

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


def execution_panels():
    report = json.loads((RES / "trained_execution.json").read_text())
    order = [
        ("softmax_memory", "softmax, with memory"),
        ("linear_memory", "linear, with memory"),
        ("linear_collapsed", "linear, no memory"),
        ("learned_threshold", "learned threshold"),
    ]
    colors = ["#1f4e79", "#5b8c5a", "#c47b2b", "#8c3a4b"]

    fig, axes = plt.subplots(3, 1, figsize=(3.35, 3.7), constrained_layout=True)

    # (1) train at p=5, evaluate at p=5 and p=10
    ax = axes[0]
    x = np.arange(len(order))
    p5 = [report[k]["one_step_median_p5"] for k, _ in order]
    p10 = []
    for k, _ in order:
        v = report[k]["one_step_median_p10"]
        p10.append(np.nan if v is None else float(v))
    width = 0.36
    ax.bar(x - width / 2, p5, width, color="#1f4e79", label="test $p=5$")
    ax.bar(x + width / 2, p10, width, color="#9bb8d3", label="test $p=10$")
    ax.axhline(1.0, color="0.35", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["softmax", "linear", "no memory", "threshold"], fontsize=7)
    ax.set_ylabel("step error", fontsize=7)
    ax.set_title("(1) train at $p{=}5$, no retraining", fontsize=8)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    ax.set_ylim(0, 20)
    ax.tick_params(labelsize=6.5)

    # (2) prompt: with vs without, against the exact block at 0
    ax = axes[1]
    names = ["with memory", "without"]
    vals = [
        report["softmax_memory"]["one_step_median_p5"],
        report["linear_collapsed"]["one_step_median_p5"],
    ]
    ax.bar(names, vals, color=["#1f4e79", "#c47b2b"])
    ax.set_ylabel("step error at $p{=}5$", fontsize=7)
    ax.set_title("(2) memory registers", fontsize=8)
    ax.tick_params(labelsize=6.5)

    # (3) distance to the exact trajectory
    ax = axes[2]
    for (k, label), color in zip(order, colors):
        curve = report[k]["unroll_p5"]
        ax.plot([s["ell"] for s in curve], [s["dist_exact"] for s in curve], color=color, label=label, lw=1.4)
    ax.set_xlabel(r"update index $\ell$", fontsize=7)
    ax.set_ylabel(r"$\|\widehat W_\ell-W_\ell\|_F$", fontsize=7)
    ax.set_yscale("log")
    ax.set_title("(3) along the update", fontsize=8)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    ax.tick_params(labelsize=6.5)

    path = FIG / "execution_panels.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


METHODS = [
    ("poly_prox", "Poly-prox"),
    ("poly_lbfgs", "Poly-LBFGS"),
    ("notears", "NOTEARS"),
    ("sort", "Sort"),
    ("pc", "PC"),
    ("ges", "GES"),
]
METHOD_COLORS = {
    "poly_prox": "#1f4e79",
    "poly_lbfgs": "#4c78a8",
    "notears": "#5b8c5a",
    "sort": "#c47b2b",
    "pc": "#8c3a4b",
    "ges": "#6b5b95",
}


def _num(text):
    if text is None or text == "":
        return np.nan
    return float(text)


def load_grid():
    import csv

    rows = []
    with (RES / "reference_grid.csv").open() as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in ("p", "degree", "m", "seed"):
                row[key] = int(row[key])
            for key in ("h", "shd", "sid", "precision", "recall", "acyclic", "varsortability"):
                row[key] = _num(row[key])
            rows.append(row)
    return rows


def mean_se(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean, se


def reference_shd(rows):
    """Mean SHD at m=10^3, the comparison in the old table."""
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.55), constrained_layout=True)
    ps = (5, 10, 20)
    x = np.arange(len(ps))
    width = 0.12
    for ax, scale, title in zip(axes, ("raw", "standardized"), ("raw", "standardized")):
        for i, (key, label) in enumerate(METHODS):
            means, ses = [], []
            for p in ps:
                chunk = [
                    r["shd"] for r in rows
                    if r["method"] == key and r["m"] == 1000 and r["scale"] == scale and r["p"] == p
                ]
                mean, se = mean_se(chunk)
                means.append(mean)
                ses.append(se)
            offset = (i - (len(METHODS) - 1) / 2) * width
            ax.bar(
                x + offset, means, width, yerr=ses, color=METHOD_COLORS[key],
                label=label, capsize=1.2,
                error_kw={"lw": 0.5, "capthick": 0.5, "ecolor": "0.2"},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"$p={p}$" for p in ps])
        ax.set_ylabel("mean SHD", fontsize=7)
        ax.set_title(title, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=6.5)
    axes[1].legend(frameon=False, ncol=2, fontsize=5.5, loc="upper left")
    path = FIG / "reference_shd.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


def reference_diagnostics(rows):
    """Main-paper checks of variance ordering and the cycle constraint."""
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 4.35), constrained_layout=True)
    m1000 = [r for r in rows if r["m"] == 1000]
    short = {
        "poly_prox": "Poly-prox",
        "poly_lbfgs": "Poly-LBFGS",
        "notears": "NOTEARS",
        "sort": "Sort",
        "pc": "PC",
        "ges": "GES",
    }

    # (1) varsortability, one value per simulated sample
    ax = axes[0]
    seen = set()
    samples = []
    for r in m1000:
        key = (r["graph"], r["p"], r["degree"], r["noise"], r["scale"], r["seed"])
        if key in seen:
            continue
        seen.add(key)
        samples.append(r)
    ps = (5, 10, 20)
    x = np.arange(len(ps))
    width = 0.36
    for shift, scale, color, label in (
        (-width / 2, "raw", "#1f4e79", "raw"),
        (width / 2, "standardized", "#c47b2b", "standardized"),
    ):
        means, ses = [], []
        for p in ps:
            mean, se = mean_se(s["varsortability"] for s in samples if s["scale"] == scale and s["p"] == p)
            means.append(mean)
            ses.append(se)
        ax.bar(x + shift, means, width, yerr=ses, color=color, label=label, capsize=1.5,
               error_kw={"lw": 0.5, "capthick": 0.5, "ecolor": "0.2"})
    ax.set_xticks(x)
    ax.set_xticklabels([f"$p={p}$" for p in ps])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("varsortability", fontsize=7)
    ax.set_title("(1) varsortability", fontsize=8)
    ax.tick_params(labelsize=6.5)

    # (2) median h
    ax = axes[1]
    solvers = [("poly_prox", "Poly-prox"), ("poly_lbfgs", "Poly-LBFGS"), ("notears", "NOTEARS")]
    x = np.arange(len(solvers))
    width = 0.36
    for shift, scale, color in ((-width / 2, "raw", "#1f4e79"), (width / 2, "standardized", "#c47b2b")):
        medians = []
        for key, _label in solvers:
            vals = np.array([
                r["h"] for r in m1000 if r["method"] == key and r["scale"] == scale
            ], dtype=float)
            vals = vals[np.isfinite(vals)]
            medians.append(float(np.median(vals)))
        ax.bar(x + shift, medians, width, color=color, label=scale)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _k, label in solvers], fontsize=6, rotation=20, ha="right")
    ax.set_ylabel(r"median $h$", fontsize=7)
    ax.set_title("(2) constraint value", fontsize=8)
    ax.legend(frameon=False, fontsize=5.5, loc="upper right")
    ax.tick_params(axis="y", labelsize=6)

    # (3) fraction acyclic
    ax = axes[2]
    x = np.arange(len(METHODS))
    width = 0.36
    for shift, scale, color in ((-width / 2, "raw", "#1f4e79"), (width / 2, "standardized", "#c47b2b")):
        means, ses = [], []
        for key, _label in METHODS:
            mean, se = mean_se(r["acyclic"] for r in m1000 if r["method"] == key and r["scale"] == scale)
            means.append(mean)
            ses.append(se)
        ax.bar(x + shift, means, width, yerr=ses, color=color, capsize=1.2,
               error_kw={"lw": 0.5, "capthick": 0.5, "ecolor": "0.2"})
    ax.set_xticks(x)
    ax.set_xticklabels([short[key] for key, _label in METHODS], rotation=40, ha="right", fontsize=5.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction acyclic", fontsize=7)
    ax.set_title("(3) stays acyclic", fontsize=8)
    ax.tick_params(axis="y", labelsize=6)

    path = FIG / "reference_diagnostics.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


def additional_solver_diagnostics(rows):
    """Appendix diagnostics of edge recovery, interventions, and sample size."""
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.35), constrained_layout=True)
    m1000 = [r for r in rows if r["m"] == 1000]
    short = {
        "poly_prox": "Poly-prox",
        "poly_lbfgs": "Poly-LBFGS",
        "notears": "NOTEARS",
        "sort": "Sort",
        "pc": "PC",
        "ges": "GES",
    }

    # (1) precision and recall at the reported slice
    ax = axes[0]
    slice_rows = [r for r in m1000 if r["p"] == 10 and r["scale"] == "raw"]
    x = np.arange(len(METHODS))
    width = 0.36
    for shift, field, color, label in (
        (-width / 2, "precision", "#1f4e79", "precision"),
        (width / 2, "recall", "#5b8c5a", "recall"),
    ):
        means, ses = [], []
        for key, _label in METHODS:
            mean, se = mean_se(r[field] for r in slice_rows if r["method"] == key)
            means.append(mean)
            ses.append(se)
        ax.bar(x + shift, means, width, yerr=ses, color=color, label=label, capsize=1.2,
               error_kw={"lw": 0.5, "capthick": 0.5, "ecolor": "0.2"})
    ax.set_xticks(x)
    ax.set_xticklabels([short[key] for key, _label in METHODS], rotation=40, ha="right", fontsize=5.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate", fontsize=7)
    ax.set_title("(1) precision, recall", fontsize=8)
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.32))
    ax.tick_params(axis="y", labelsize=6)

    # (2) SID on raw Gaussian p=10, where the estimate is a DAG
    ax = axes[1]
    sid_methods = [
        ("poly_prox", "Poly-prox"),
        ("poly_lbfgs", "Poly-LBFGS"),
        ("notears", "NOTEARS"),
        ("sort", "Sort"),
    ]
    means, ses = [], []
    for key, _label in sid_methods:
        chunk = [
            r["sid"] for r in rows
            if r["method"] == key and r["p"] == 10 and r["m"] == 1000
            and r["scale"] == "raw" and r["noise"] == "gaussian"
        ]
        mean, se = mean_se(chunk)
        means.append(mean)
        ses.append(se)
    ax.bar(
        [label for _k, label in sid_methods], means, yerr=ses,
        color=[METHOD_COLORS[k] for k, _l in sid_methods], capsize=1.5,
        error_kw={"lw": 0.5, "capthick": 0.5, "ecolor": "0.2"},
    )
    ax.set_ylabel("mean SID", fontsize=7)
    ax.set_title(r"(2) SID, $p{=}10$", fontsize=8)
    ax.tick_params(axis="x", labelrotation=30, labelsize=5.5)
    ax.tick_params(axis="y", labelsize=6)

    # (3) SHD against sample size, raw, so the m axis of the grid is visible
    ax = axes[2]
    # Degree 2, Gaussian, both graphs, three seeds: the cells shared by every m.
    ms = (100, 1000, 10000)
    for key, label in METHODS:
        ys = []
        for m in ms:
            chunk = [
                r["shd"] for r in rows
                if r["method"] == key and r["m"] == m and r["scale"] == "raw"
                and r["degree"] == 2 and r["noise"] == "gaussian" and r["p"] in (5, 10)
            ]
            mean, _se = mean_se(chunk)
            ys.append(mean)
        ax.plot(ms, ys, marker="o", ms=3.5, lw=1.2, color=METHOD_COLORS[key], label=label)
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels([r"$10^2$", r"$10^3$", r"$10^4$"])
    ax.set_xlabel("samples $m$", fontsize=7)
    ax.set_ylabel("mean SHD", fontsize=7)
    ax.set_title("(3) more samples", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=6)
    ax.legend(frameon=False, fontsize=4.8, ncol=2, loc="center", bbox_to_anchor=(0.5, 0.46))
    path = FIG / "additional_solver_diagnostics.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


def exact_checks():
    report = json.loads((RES / "exact_execution.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(3.35, 3.45), constrained_layout=True)

    ax = axes[0, 0]
    # float64 and squaring_vs_power sit within floating-point noise of each
    # other (both ~1e-16), so a plain solid line hides one under the other;
    # float64 gets a dashed style, a distinct marker, and a higher zorder so
    # both remain visible where they nearly coincide.
    dtype_style = (
        ("float64", "float64", "#1f4e79", "--", "s", 3),
        ("float32", "float32", "#5b8c5a", "-", "o", 2),
        ("bfloat16", "bfloat16", "#c47b2b", "-", "o", 2),
        ("squaring_vs_power", "squaring", "#8c3a4b", "-", "o", 2),
    )
    for key, label, color, ls, marker, zorder in dtype_style:
        chunk = [r for r in report["summary"] if r["dtype"] == key]
        chunk = sorted(chunk, key=lambda r: r["p"])
        ax.plot([r["p"] for r in chunk], [r["max_W"] for r in chunk], marker=marker,
                 ms=3.5, lw=1.3, ls=ls, color=color, label=label, zorder=zorder)
    ax.set_yscale("log")
    ax.set_xticks([3, 5, 10, 20])
    ax.set_xlabel("$p$")
    ax.set_ylabel(r"max $\|W\|$ residual", fontsize=7)
    ax.set_title("(1) one update", fontsize=8)
    # float32 sits near 1e-7 and float64 near 1e-16, so the legend goes in that gap.
    ax.legend(frameon=False, fontsize=5, loc="center", bbox_to_anchor=(0.62, 0.32),
              handlelength=1.4, labelspacing=0.2)
    ax.tick_params(labelsize=6)

    ax = axes[0, 1]
    mult = report["multiplier"]
    t = np.array([r["t"] for r in mult])
    c = np.array([r["c"] for r in mult])
    ax.plot(t, c, color="#1f4e79", lw=1.4, label=r"$c$")
    ax.plot(t, 0.4 / t ** 3, color="#c47b2b", lw=1.1, ls="--", label=r"$(r-\lambda)t^{-3}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$c$", fontsize=7)
    ax.set_title(r"(2) multiplier", fontsize=8)
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    ax.tick_params(labelsize=6)

    ax = axes[1, 0]
    rel = [r for r in report["relative_accuracy"] if abs(r["nu"] - 0.1) < 1e-12]
    rel = sorted(rel, key=lambda r: r["t"])
    ts = [r["t"] for r in rel]
    ax.plot(ts, [r["relative_grad_error"] for r in rel], color="#1f4e79", marker="o", ms=3.0, lw=1.2, label="relative")
    ax.plot(ts, [r["bound"] for r in rel], color="#1f4e79", ls="--", lw=1.0, label="bound")
    ax.plot(ts, [r["absolute_grad_error"] for r in rel], color="#c47b2b", marker="o", ms=3.0, lw=1.2, label="absolute")
    ax.set_yscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel("gradient error", fontsize=7)
    ax.set_title("(3) gradient error", fontsize=8)
    ax.legend(frameon=False, fontsize=5.5, loc="upper right",
              bbox_to_anchor=(0.99, 0.99), borderaxespad=0)
    ax.tick_params(labelsize=6)

    ax = axes[1, 1]
    hist = report["alpha_drift"]["history"]
    ell = np.array([r["ell"] + 1 for r in hist])
    ax.plot(ell, [r["predicted_alpha_error"] for r in hist], color="#1f4e79", lw=1.3, label=r"$\ell\cdot 10^{-3}$")
    ax.plot(ell, [abs(r["hat_alpha"]) for r in hist], color="#1f4e79", ls="none", marker="o", ms=3.0)
    ax.plot(ell, [r["exact_W"] for r in hist], color="#5b8c5a", lw=1.3, label="exact $W$")
    ax.plot(ell, [r["hat_W"] for r in hist], color="#c47b2b", lw=1.1, ls="--", label="perturbed $W$")
    ax.set_xlabel(r"step $\ell$")
    ax.set_ylabel("error", fontsize=7)
    ax.set_title("(4) drift at the origin", fontsize=8)
    # The two $W$ traces lie on the axis, and the multiplier error is the diagonal,
    # so a boxed legend cannot sit in either corner. Label the traces directly.
    ax.text(0.28, 0.78, r"$\ell\cdot 10^{-3}$", color="#1f4e79", fontsize=5.5,
            transform=ax.transAxes, ha="left", va="center")
    ax.text(0.62, 0.20, r"$W = 0$", color="#5b8c5a", fontsize=5.5,
            transform=ax.transAxes, ha="left", va="center")
    ax.tick_params(labelsize=6)

    path = FIG / "exact_checks.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    execution_panels()
    grid = load_grid()
    reference_shd(grid)
    reference_diagnostics(grid)
    additional_solver_diagnostics(grid)
    exact_checks()
