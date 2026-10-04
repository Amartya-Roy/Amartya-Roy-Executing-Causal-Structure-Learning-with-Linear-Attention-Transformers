"""External-topology linear-SEM benchmark.

The graphs are published Bayesian-network topologies. The observations
are not the original discrete tables. Each topology gets one weighted
adjacency matrix from the protocol below, and every method sees the
same centered Gaussian samples.

Frozen before any recovery score on these graphs:

- Topologies: every bnlearn repository network with 5 <= p <= 20,
  namely asia, cancer, earthquake, survey, sachs, and child.
  CHILD is the one drawn in the figure. It was chosen because it is
  the standard 20-node network, not because of a score.
- Edge weights: independent Uniform[0.5, 2] magnitudes and independent
  random signs. One draw per topology. The generator is
  SeedSequence([1000 + index, 1]), with index the position of the
  topology in TOPOLOGIES.
- Noise: equal-variance Gaussian, variance 1, raw scale, then centered.
  No column standardization. Data seed
  SeedSequence([2000 + index, m, draw]).
- Sample sizes: 1000, 5000, 10000 on every topology, plus 500 on CHILD.
- Ten independent noise draws at each sample size.
- Poly-prox, Poly-LBFGS, and NOTEARS use lambda = 0.1.
  The edge threshold is 0.3. Both are the values already used in the
  paper. They are not retuned on these graphs.
- DAGMA uses its package penalty lambda_1 = 0.03. The same threshold
  0.3 is applied after the fit.
- GOLEM-EV uses lambda_1 = 0.02, lambda_2 = 5, and 2e4 Adam steps.
- The heatmap is the CHILD draw at m = 10000 whose Poly-prox SHD is
  closest to the median of those ten draws. Ties take the smaller draw
  index. That is a typical draw, not the best one.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

from experiments.baselines import (
    dagma_estimate,
    ges_estimate,
    golem_ev,
    notears_expm,
    notears_poly_lbfgs,
    pc_estimate,
    sortnregress,
)
from experiments.hgrad import h_and_grad
from experiments.metrics import binarize, is_acyclic, shd
from experiments.realization import one_update
from experiments.run_additional import LAM, THRESHOLD, _norms, graph_metrics
from experiments.sem import covariance, polynomial_degree, simulate_sem
from experiments.solver import proximal_al

ROOT = Path(__file__).resolve().parents[1]
BN = Path(__file__).resolve().parent / "data" / "bn"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

# Declared order. CHILD is last so a partial file still has the smaller graphs,
# and the figure topology is the preferred 20-node network.
TOPOLOGIES = ("asia", "cancer", "earthquake", "survey", "sachs", "child", "alarm")
SAMPLE_SIZES = (1000, 5000, 10000)
CHILD_EXTRA = (500,)
N_DRAWS = 10
GOLEM_STEPS = 20000
WEIGHT_LO, WEIGHT_HI = 0.5, 2.0


def parse_bif(path: Path):
    text = path.read_text()
    nodes = re.findall(r"^variable\s+(\S+)\s*\{", text, re.M)
    edges = []
    for child, parents in re.findall(
        r"probability\s*\(\s*([^)|]+?)(?:\s*\|\s*([^)]+))?\s*\)", text
    ):
        if not parents:
            continue
        for parent in parents.split(","):
            edges.append((parent.strip(), child.strip()))
    index = {name: i for i, name in enumerate(nodes)}
    missing = [n for e in edges for n in e if n not in index]
    if missing:
        raise ValueError(f"{path.name}: unknown nodes {missing[:5]}")
    W = np.zeros((len(nodes), len(nodes)))
    for parent, child in edges:
        W[index[parent], index[child]] = 1.0
    if not is_acyclic(W):
        raise ValueError(f"{path.name} is not a DAG")
    return nodes, W


def load_topologies():
    graphs = {}
    for name in TOPOLOGIES:
        nodes, binary = parse_bif(BN / f"{name}.bif")
        graphs[name] = {"nodes": nodes, "binary": binary, "n_edges": int(binary.sum())}
    return graphs


def _index(name: str) -> int:
    return TOPOLOGIES.index(name)


def assign_weights(binary: np.ndarray, name: str) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([1000 + _index(name), 1]))
    W = np.zeros_like(binary)
    for i, j in zip(*np.nonzero(binary)):
        sign = float(rng.choice([-1.0, 1.0]))
        W[i, j] = sign * float(rng.uniform(WEIGHT_LO, WEIGHT_HI))
    return W


def replay_trajectory(Sigma, trace, alpha_log):
    """Replay every accepted proximal step and every multiplier update.

    The arithmetic is the realization's squaring schedule in float64 NumPy,
    the same operations the solver uses for h and its gradient. A torch
    matmul differs by about one ulp, and on this trajectory that ulp is
    amplified once the multiplier grows, so the open loop is kept in NumPy.
    """
    from collections import defaultdict

    p = Sigma.shape[0]
    N = polynomial_degree(p)
    mask = np.ones((p, p))
    np.fill_diagonal(mask, 0.0)
    pending = defaultdict(list)
    for event in alpha_log:
        pending[int(event["n_steps"])].append(event)
    W = np.zeros((p, p))
    worst_f = 0.0
    worst_max = 0.0
    alpha = 0.0
    alpha_gap = 0.0
    h_gap = 0.0

    def apply_alpha(n_steps):
        nonlocal alpha, alpha_gap, h_gap
        for event in pending.pop(n_steps, []):
            h_hat, _ = h_and_grad(W, N)
            alpha = alpha + event["rho"] * float(h_hat)
            alpha_gap = max(alpha_gap, abs(alpha - event["alpha"]))
            h_gap = max(h_gap, abs(float(h_hat) - event["h"]))

    apply_alpha(0)
    for i, step in enumerate(trace, start=1):
        W, _, _, _ = one_update(
            Sigma, W, mask, step["alpha"], step["rho"], step["gamma"], LAM, 0.0, N
        )
        gap = _norms(W, step["W_out"])
        worst_f = max(worst_f, gap["frobenius"])
        worst_max = max(worst_max, gap["max"])
        apply_alpha(i)
    final = _norms(W, trace[-1]["W_out"]) if trace else {"frobenius": 0.0, "max": 0.0}
    return W, final, worst_f, worst_max, alpha_gap, h_gap


def support_disagreement(A, B):
    """SHD between two estimates, with the same threshold on both."""
    return int(shd(binarize(A, THRESHOLD).astype(float), B, THRESHOLD))


def _h(W):
    N = polynomial_degree(W.shape[0])
    value, _ = h_and_grad(W, N)
    return float(value)


def run_one(name, true_W, m, draw):
    rng = np.random.default_rng(np.random.SeedSequence([2000 + _index(name), int(m), int(draw)]))
    X = simulate_sem(true_W, m, "gaussian", rng, standardize=False)
    Sigma = covariance(X)
    trace = []
    alpha_log = []
    t0 = time.perf_counter()
    solved = proximal_al(Sigma, lam=LAM, kind="poly", trace=trace, alpha_log=alpha_log)
    prox_seconds = time.perf_counter() - t0
    W_ref = solved["W"]
    W_hat, final, worst_f, worst_max, alpha_gap, h_gap = replay_trajectory(Sigma, trace, alpha_log)
    methods = {
        "poly_prox": (W_ref, prox_seconds, float(solved["h"])),
        "transformer": (W_hat, 0.0, _h(W_hat)),
    }
    runners = {
        "poly_lbfgs": lambda: notears_poly_lbfgs(Sigma, lam=LAM),
        "notears": lambda: notears_expm(Sigma, lam=LAM),
        "dagma": lambda: dagma_estimate(X, threshold=0.0),
        "golem": lambda: golem_ev(X, num_iter=GOLEM_STEPS, seed=draw),
        "sort": lambda: sortnregress(X),
        "pc": lambda: pc_estimate(X),
        "ges": lambda: ges_estimate(X),
    }
    scores = {}
    for key, (W, seconds, h_value) in methods.items():
        metrics = graph_metrics(true_W, W)
        metrics["seconds"] = float(seconds)
        metrics["h"] = float(h_value)
        scores[key] = metrics
    for key, fn in runners.items():
        t0 = time.perf_counter()
        est = fn()
        seconds = time.perf_counter() - t0
        if isinstance(est, dict):
            h_value = float(est.get("h", _h(est["W"])))
            est = est["W"]
        else:
            est = np.asarray(est, dtype=np.float64)
            h_value = _h(est)
        metrics = graph_metrics(true_W, est)
        metrics["seconds"] = float(seconds)
        metrics["h"] = float(h_value)
        scores[key] = metrics
        methods[key] = (est, seconds, h_value)
    row = {
        "topology": name,
        "m": int(m),
        "draw": int(draw),
        "n_steps": len(trace),
        "exec_final": final,
        "exec_worst": {"frobenius": worst_f, "max": worst_max},
        "alpha_gap": float(alpha_gap),
        "h_gap": float(h_gap),
        "support_shd": support_disagreement(W_hat, W_ref),
        "solver_error": _norms(W_ref, true_W),
        "total_error": _norms(W_hat, true_W),
        "scores": scores,
    }
    saved = {
        "poly_prox": W_ref,
        "transformer": W_hat,
        "notears": methods["notears"][0],
    }
    return row, saved


def main():
    graphs = load_topologies()
    only = os.environ.get("ONLY", "")
    names = tuple(n for n in only.split(",") if n) or TOPOLOGIES
    for name, graph in graphs.items():
        print(f"{name}: p={len(graph['nodes'])} edges={graph['n_edges']}", flush=True)
    weights = {name: assign_weights(graph["binary"], name) for name, graph in graphs.items()}
    path = OUT / ("external_alarm.json" if only else "external.json")
    payload = {"protocol": {
        "topologies": list(TOPOLOGIES),
        "weight": [WEIGHT_LO, WEIGHT_HI],
        "lambda": LAM,
        "threshold": THRESHOLD,
        "draws": N_DRAWS,
        "golem_steps": GOLEM_STEPS,
    }, "edges": {name: int(graph["n_edges"]) for name, graph in graphs.items()},
        "p": {name: len(graph["nodes"]) for name, graph in graphs.items()}}
    if path.exists():
        old = json.loads(path.read_text())
        payload["rows"] = old.get("rows", [])
    else:
        payload["rows"] = []
    done = {(r["topology"], r["m"], r["draw"]) for r in payload["rows"]}
    figure = {}
    figure_path = OUT / "external_child_m10000.npz"
    if figure_path.exists():
        figure = dict(np.load(figure_path))
    for name in names:
        sizes = SAMPLE_SIZES + (CHILD_EXTRA if name == "child" else ())
        for m in sizes:
            for draw in range(N_DRAWS):
                key = (name, int(m), int(draw))
                if key in done:
                    continue
                t0 = time.perf_counter()
                row, saved = run_one(name, weights[name], m, draw)
                payload["rows"].append(row)
                path.write_text(json.dumps(payload))
                if name == "child" and int(m) == 10000:
                    figure["truth"] = weights[name]
                    figure[f"poly_{draw}"] = saved["poly_prox"]
                    figure[f"tr_{draw}"] = saved["transformer"]
                    figure[f"nt_{draw}"] = saved["notears"]
                    np.savez(figure_path, **figure)
                prox = row["scores"]["poly_prox"]["shd"]
                note = row["scores"]["notears"]["shd"]
                print(
                    f"{name} m={m} draw={draw} prox={prox} notears={note} "
                    f"exec={row['exec_final']['frobenius']:.2e} "
                    f"sec={time.perf_counter() - t0:.1f}",
                    flush=True,
                )
    print("wrote", path)


def _median_draw(rows, name, m=10000):
    """Draw whose Poly-prox SHD is closest to the median. Ties take the smaller index."""
    chunk = [r for r in rows if r["topology"] == name and int(r["m"]) == int(m)]
    if not chunk:
        raise KeyError(f"no rows for {name} at m={m}")
    median = float(np.median([r["scores"]["poly_prox"]["shd"] for r in chunk]))
    chosen = min(chunk, key=lambda r: (abs(r["scores"]["poly_prox"]["shd"] - median), r["draw"]))
    return int(chosen["draw"]), median, int(chosen["scores"]["poly_prox"]["shd"])


def save_median_graphs(m=10000):
    """One typical draw per topology, Poly-prox and its transformer replay only.

    The draw is fixed by the median Poly-prox SHD already stored in the
    result files. Other baselines are not rerun.
    """
    rows = []
    for filename in ("external.json", "external_alarm.json"):
        path = OUT / filename
        if path.exists():
            rows.extend(json.loads(path.read_text()).get("rows", []))
    graphs = load_topologies()
    weights = {name: assign_weights(graph["binary"], name) for name, graph in graphs.items()}
    saved = {}
    for name in TOPOLOGIES:
        draw, median, shd_value = _median_draw(rows, name, m)
        true_W = weights[name]
        rng = np.random.default_rng(np.random.SeedSequence([2000 + _index(name), int(m), int(draw)]))
        X = simulate_sem(true_W, m, "gaussian", rng, standardize=False)
        Sigma = covariance(X)
        trace, alpha_log = [], []
        solved = proximal_al(Sigma, lam=LAM, kind="poly", trace=trace, alpha_log=alpha_log)
        W_hat, final, _, _, _, _ = replay_trajectory(Sigma, trace, alpha_log)
        got = int(shd(true_W, solved["W"], THRESHOLD))
        if got != shd_value:
            raise RuntimeError(f"{name} draw {draw}: SHD {got} != stored {shd_value}")
        saved[f"{name}_truth"] = true_W
        saved[f"{name}_poly"] = solved["W"]
        saved[f"{name}_transformer"] = W_hat
        saved[f"{name}_draw"] = np.array([draw])
        saved[f"{name}_shd"] = np.array([shd_value])
        print(
            f"{name} draw={draw} median={median:.1f} shd={shd_value} "
            f"exec={final['frobenius']:.2e}",
            flush=True,
        )
    path = OUT / "inherit_graphs.npz"
    np.savez(path, **saved)
    print("wrote", path)


if __name__ == "__main__":
    main()
