"""Experiment 1: numerical reference on simulated linear SEMs.

The grid is the factor list in the planned evaluation, with two cuts that
keep the sweep finite: p stops at 20 (not 50) and the m=10^4 corner is run
only for p in {5, 10}, degree 2, Gaussian noise. Three seeds per cell.
"""

from __future__ import annotations

import csv
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

from experiments.baselines import ges_estimate, notears_expm, notears_poly_lbfgs, pc_estimate, sortnregress
from experiments.hgrad import h_and_grad
from experiments.metrics import directed_precision_recall, is_acyclic, binarize, shd, sid, varsortability
from experiments.sem import covariance, polynomial_degree, sample_dag, simulate_sem
from experiments.solver import proximal_al, score

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
CSV_PATH = OUT / "reference_grid.csv"

THRESHOLD = 0.3
LAM = 0.1

FIELDS = [
    "graph", "p", "degree", "noise", "m", "scale", "seed", "method",
    "seconds", "h", "score", "G_gamma", "shd", "sid", "precision", "recall",
    "acyclic", "n_true_edges", "n_est_edges", "varsortability", "error",
]


def cells():
    graphs = ("er", "sf")
    degrees = (1, 2, 4)
    noises = ("gaussian", "laplace")
    scales = ("raw", "standardized")
    seeds = (0, 1, 2)
    for graph in graphs:
        for degree in degrees:
            for noise in noises:
                for scale in scales:
                    for seed in seeds:
                        for p, m in (
                            (5, 100), (5, 1000), (5, 10000),
                            (10, 100), (10, 1000), (10, 10000),
                            (20, 100), (20, 1000),
                        ):
                            # The large-sample corner is restricted; see the module docstring.
                            if m == 10000 and not (degree == 2 and noise == "gaussian"):
                                continue
                            yield graph, p, degree, noise, m, scale, seed


def _empty_row(**kwargs):
    row = {k: "" for k in FIELDS}
    row.update(kwargs)
    return row


def evaluate_weighted(true_W, X, est_W, Sigma):
    N = polynomial_degree(true_W.shape[0])
    h, _ = h_and_grad(est_W, N)
    prec, rec = directed_precision_recall(true_W, est_W, THRESHOLD)
    binary = binarize(est_W, THRESHOLD)
    return {
        "h": h,
        "score": score(Sigma, est_W, LAM),
        "shd": shd(true_W, est_W, THRESHOLD),
        "sid": sid(true_W, est_W, THRESHOLD),
        "precision": prec,
        "recall": rec,
        "acyclic": int(is_acyclic(binary)),
        "n_est_edges": int(binary.sum()),
        "G_gamma": "",
    }


def evaluate_cpdag(true_W, est_A):
    prec, rec = directed_precision_recall(true_W, est_A, 0.0)
    return {
        "h": "",
        "score": "",
        "shd": shd(true_W, est_A, 0.0),
        "sid": sid(true_W, est_A, 0.0),
        "precision": prec,
        "recall": rec,
        "acyclic": int(is_acyclic(binarize(est_A, 0.0))),
        "n_est_edges": int(binarize(est_A, 0.0).sum()),
        "G_gamma": "",
    }


def run_cell(graph, p, degree, noise, m, scale, seed):
    # Scale is applied after sampling, so raw and standardized share the draw.
    # Sample size is not part of the seed: the m=100 series is the prefix of m=1000.
    rng = np.random.default_rng(
        1_000_003 * seed + 17 * p + int(degree) + (0 if graph == "er" else 97)
        + (1 if noise == "gaussian" else 2) * 1_000_000
    )
    true_W = sample_dag(p, degree, graph, rng)
    X = simulate_sem(true_W, m, noise, rng, standardize=False)
    if scale == "standardized":
        std = X.std(axis=0, ddof=0)
        std[std < 1e-12] = 1.0
        X = X / std
    Sigma = covariance(X)
    vsort = varsortability(X, true_W)
    n_true = int(np.sum(np.abs(true_W) > 0))
    base = dict(
        graph=graph, p=p, degree=degree, noise=noise, m=m, scale=scale, seed=seed,
        n_true_edges=n_true, varsortability=vsort,
    )
    rows = []

    jobs = (
        ("poly_prox", lambda: proximal_al(Sigma, lam=LAM, kind="poly")),
        ("poly_lbfgs", lambda: notears_poly_lbfgs(Sigma, lam=LAM)),
        ("notears", lambda: notears_expm(Sigma, lam=LAM)),
        ("sort", lambda: sortnregress(X)),
        ("pc", lambda: pc_estimate(X)),
        ("ges", lambda: ges_estimate(X)),
    )
    for name, fn in jobs:
        t0 = time.time()
        try:
            out = fn()
            seconds = time.time() - t0
            if name in ("pc", "ges"):
                metrics = evaluate_cpdag(true_W, out)
                metrics["G_gamma"] = ""
            elif name == "sort":
                metrics = evaluate_weighted(true_W, X, out, Sigma)
                metrics["G_gamma"] = ""
            else:
                metrics = evaluate_weighted(true_W, X, out["W"], Sigma)
                metrics["h"] = out["h"]
                metrics["score"] = out["score"]
                metrics["G_gamma"] = out["G_gamma"]
            sid_val = metrics["sid"]
            metrics["sid"] = "" if sid_val is None else sid_val
            rows.append(_empty_row(**base, method=name, seconds=round(seconds, 4), error="", **metrics))
        except Exception as exc:
            rows.append(_empty_row(
                **base, method=name, seconds=round(time.time() - t0, 4),
                error=f"{type(exc).__name__}: {exc}",
            ))
            traceback.print_exc()
    return rows


def done_keys(path: Path):
    if not path.exists():
        return set()
    keys = set()
    with path.open() as f:
        for row in csv.DictReader(f):
            keys.add((row["graph"], int(row["p"]), int(row["degree"]), row["noise"], int(row["m"]), row["scale"], int(row["seed"]), row["method"]))
    return keys


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    write_header = not CSV_PATH.exists()
    finished = done_keys(CSV_PATH)
    pending = []
    for cell in cells():
        # A cell is done when all six methods are present.
        methods = ("poly_prox", "poly_lbfgs", "notears", "sort", "pc", "ges")
        if all((cell[0], cell[1], cell[2], cell[3], cell[4], cell[5], cell[6], method) in finished for method in methods):
            continue
        pending.append(cell)
    print(f"{len(pending)} cells to run, {len(finished)} method-rows already stored")
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for i, cell in enumerate(pending, start=1):
            t0 = time.time()
            rows = run_cell(*cell)
            for row in rows:
                writer.writerow(row)
            f.flush()
            shds = ", ".join(f"{r['method']}={r['shd']}" for r in rows)
            print(f"[{i}/{len(pending)}] {cell} in {time.time()-t0:.1f}s  {shds}", flush=True)


if __name__ == "__main__":
    main()
