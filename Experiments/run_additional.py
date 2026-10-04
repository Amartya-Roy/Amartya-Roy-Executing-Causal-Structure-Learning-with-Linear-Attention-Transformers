"""Additional checks that the main grid does not cover.

Five pieces, kept separate on purpose:

1. Error decomposition. On each synthetic draw, compare the true graph,
   the converged Poly-prox estimate, a 10-step Poly-prox iterate, and the
   fixed-weight block run on those same 10 steps.
2. Raw samples versus a precomputed covariance. The block's linear head
   forms Sigma from centered rows; the proximal controls stay fixed.
3. Sachs protein data. One observational table, not a claim that the
   linear equal-noise model holds there.
4. Execution at p_max in {16, 32, 64, 128}. Random states, not a full
   causal benchmark.
5. DAGMA and GOLEM on the degree-2, m=1000 slice, beside the solvers
   already in the paper. Same threshold 0.3. Their penalties stay at the
   authors' defaults, because those penalties are not the paper's lambda.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.baselines import (
    dagma_estimate,
    ges_estimate,
    golem_ev,
    notears_expm,
    notears_poly_lbfgs,
    pc_estimate,
    sortnregress,
)
from experiments.hgrad import h_and_grad, h_and_grad_power
from experiments.metrics import (
    binarize,
    directed_precision_recall,
    is_acyclic,
    shd,
    sid,
)
from experiments.realization import one_update_reference, pad_problem
from experiments.run_exact import one_update_torch
from experiments.sem import covariance, polynomial_degree, sample_dag, simulate_sem
from experiments.solver import proximal_al

OUT = Path(__file__).resolve().parent / "results"
DATA = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

LAM = 0.1
THRESHOLD = 0.3
FINITE_STEPS = 10
GOLEM_STEPS = 20000

# bnlearn's 17-arc Sachs network, in the column order of the cytometric table:
# praf, pmek, plcg, PIP2, PIP3, p44/42, PKA is index 7, PKC index 8.
# An entry (i, j) is the edge i -> j.
SACHS_EDGES = (
    (0, 1),   # praf -> pmek
    (7, 1),   # PKA -> pmek
    (8, 1),   # PKC -> pmek
    (2, 3),   # plcg -> PIP2
    (4, 3),   # PIP3 -> PIP2
    (2, 4),   # plcg -> PIP3
    (1, 5),   # pmek -> erk
    (7, 5),   # PKA -> erk
    (5, 6),   # erk -> akt
    (7, 6),   # PKA -> akt
    (8, 7),   # PKC -> PKA
    (2, 8),   # plcg -> PKC
    (3, 8),   # PIP2 -> PKC
    (7, 9),   # PKA -> P38
    (8, 9),   # PKC -> P38
    (7, 10),  # PKA -> jnk
    (8, 10),  # PKC -> jnk
)


def _f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _mask(p):
    mask = np.ones((p, p))
    np.fill_diagonal(mask, 0.0)
    return mask


def _norms(A, B):
    D = np.asarray(A, dtype=np.float64) - np.asarray(B, dtype=np.float64)
    return {
        "frobenius": float(np.linalg.norm(D)),
        "max": float(np.max(np.abs(D))) if D.size else 0.0,
    }


def sigma_head(X, dtype):
    """One linear head: Sigma = X^T X / m, in `dtype`, returned as float64."""
    tokens = torch.tensor(np.asarray(X, dtype=np.float64), dtype=dtype)
    m = tokens.shape[0]
    gram = (tokens.T @ tokens) / tokens.new_tensor(float(m))
    return gram.detach().to(torch.float64).cpu().numpy()


def replay(Sigma, trace, n_steps, dtype):
    """Run the recorded proximal controls through the fixed-weight arithmetic."""
    p = Sigma.shape[0]
    N = polynomial_degree(p)
    mask = _mask(p)
    W = np.zeros((p, p))
    steps = trace[:n_steps]
    for step in steps:
        Sp, Wp, Mp = pad_problem(Sigma, W, mask, N)
        W_next, _, _, _ = one_update_torch(
            Sp, Wp, Mp, step["alpha"], step["rho"], step["gamma"], LAM, 0.0, N, dtype
        )
        W = np.asarray(W_next[:p, :p], dtype=np.float64)
    return W


def graph_metrics(true_W, est_W):
    precision, recall = directed_precision_recall(true_W, est_W, THRESHOLD)
    binary = binarize(est_W, THRESHOLD)
    acyclic = bool(is_acyclic(binary))
    sid_value = sid(true_W, est_W, THRESHOLD) if acyclic and is_acyclic(binarize(true_W, 0.0)) else None
    return {
        "shd": int(shd(true_W, est_W, THRESHOLD)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": _f1(precision, recall),
        "acyclic": acyclic,
        "sid": None if sid_value is None else float(sid_value),
    }


def _instance(p, m, seed, graph="er", noise="gaussian", scale="raw", degree=2.0, weight_scale=(0.5, 2.0)):
    rng = np.random.default_rng([p, m, seed, {"er": 0, "sf": 1}[graph], {"gaussian": 0, "laplace": 1}[noise], {"raw": 0, "standardized": 1}[scale], int(10 * degree)])
    true_W = sample_dag(p, degree, graph, rng, weight_scale=weight_scale)
    X = simulate_sem(true_W, m, noise, rng, standardize=(scale == "standardized"))
    return true_W, X, covariance(X)


def run_decomposition_and_raw():
    """Shared synthetic draws for the error split and the raw-sample head."""
    rows = []
    specs = []
    for p in (5, 10, 20):
        for m in (100, 1000):
            for seed in (0, 1, 2):
                specs.append((p, m, seed))
    for i, (p, m, seed) in enumerate(specs, start=1):
        t0 = time.perf_counter()
        true_W, X, Sigma = _instance(p, m, seed)
        trace = []
        solved = proximal_al(Sigma, lam=LAM, kind="poly", trace=trace)
        W_ref = solved["W"]
        n_full = len(trace)
        n_finite = min(FINITE_STEPS, n_full)
        W_L = trace[n_finite - 1]["W_out"]
        W_hat = replay(Sigma, trace, n_finite, torch.float64)
        W_hat_full = replay(Sigma, trace, n_full, torch.float64)
        W_hat32 = replay(Sigma, trace, n_full, torch.float32)

        Sigma64 = sigma_head(X, torch.float64)
        Sigma32 = sigma_head(X, torch.float32)
        W_raw64 = replay(Sigma64, trace, n_full, torch.float64)
        W_raw64_one = replay(Sigma64, trace, 1, torch.float64)
        W_sig_one = replay(Sigma, trace, 1, torch.float64)
        W_raw32 = replay(Sigma32, trace, n_full, torch.float32)
        W_raw32_one = replay(Sigma32, trace, 1, torch.float32)
        W_sig32_one = replay(Sigma, trace, 1, torch.float32)
        support_64 = shd(W_hat_full, W_raw64, THRESHOLD)
        support_32 = shd(replay(Sigma, trace, n_full, torch.float32), W_raw32, THRESHOLD)

        row = {
            "p": p,
            "m": m,
            "seed": seed,
            "n_steps": n_full,
            "n_finite": n_finite,
            "h_ref": float(solved["h"]),
            "seconds": float(time.perf_counter() - t0),
            "exec_finite": _norms(W_hat, W_L),
            "exec_full": _norms(W_hat_full, W_ref),
            "exec_full_float32": _norms(W_hat32, W_ref),
            "opt": _norms(W_L, W_ref),
            "stat": _norms(W_ref, true_W),
            "total": _norms(W_hat, true_W),
            "graph_ref": graph_metrics(true_W, W_ref),
            "sigma_head_float64": _norms(Sigma64, Sigma),
            "sigma_head_float32": _norms(Sigma32, Sigma),
            "raw_one_float64": _norms(W_raw64_one, W_sig_one),
            "raw_full_float64": _norms(W_raw64, W_hat_full),
            "raw_one_float32": _norms(W_raw32_one, W_sig32_one),
            "raw_full_float32": _norms(W_raw32, replay(Sigma, trace, n_full, torch.float32)),
            "raw_support_shd_float64": int(support_64),
            "raw_support_shd_float32": int(support_32),
        }
        rows.append(row)
        print(
            f"decomp {i}/{len(specs)} p={p} m={m} seed={seed} "
            f"steps={n_full} exec={row['exec_full']['frobenius']:.2e} "
            f"opt={row['opt']['frobenius']:.2e} stat={row['stat']['frobenius']:.2e} "
            f"sec={row['seconds']:.1f}",
            flush=True,
        )
    return rows


def run_scaling():
    rows = []
    rng = np.random.default_rng(7)
    dtypes = {"float64": torch.float64, "float32": torch.float32}
    for p_max in (16, 32, 64, 128):
        N = polynomial_degree(p_max)
        s = int(np.log2(N))
        layers = 2 * s + 8
        width = 13 * p_max + 18
        sizes = (max(2, p_max // 8), p_max // 2, p_max)
        for p in sizes:
            for name, dt in dtypes.items():
                w_res, max_res, h_res, g_res, times = [], [], [], [], []
                for _state in range(4):
                    Sigma, W, mask, alpha, rho, gamma, lam, b = _random_state(p, rng)
                    t0 = time.perf_counter()
                    Sp, Wp, Mp = pad_problem(Sigma, W, mask, p_max)
                    got_W, _, got_h, _ = one_update_torch(
                        Sp, Wp, Mp, alpha, rho, gamma, lam, b, N, dt
                    )
                    times.append(time.perf_counter() - t0)
                    ref_W, _, ref_h, _ = one_update_reference(
                        Sigma, W, mask, alpha, rho, gamma, lam, b, N
                    )
                    active = got_W[:p, :p]
                    w_res.append(float(np.linalg.norm(active - ref_W)))
                    max_res.append(float(np.max(np.abs(active - ref_W))))
                    h_res.append(float(abs(got_h - ref_h)))
                    h_sq, g_sq = h_and_grad(W, N)
                    h_pw, g_pw = h_and_grad_power(W, N)
                    g_res.append(float(np.linalg.norm(g_sq - g_pw)))
                    h_res[-1] = max(h_res[-1], abs(h_sq - h_pw))
                rows.append({
                    "p_max": p_max,
                    "p": p,
                    "dtype": name,
                    "layers": layers,
                    "width": width,
                    "stream_bytes": int((p_max + 1) * width * (8 if name == "float64" else 4)),
                    "W_residual_max": float(np.max(w_res)),
                    "W_residual_median": float(np.median(w_res)),
                    "max_entry": float(np.max(max_res)),
                    "h_error_max": float(np.max(h_res)),
                    "grad_h_error_max": float(np.max(g_res)),
                    "seconds_median": float(np.median(times)),
                })
                print(
                    f"scale p_max={p_max} p={p} {name} "
                    f"W={rows[-1]['W_residual_max']:.2e} layers={layers} "
                    f"sec={rows[-1]['seconds_median']:.3f}",
                    flush=True,
                )
    return rows


def _random_state(p, rng):
    W_true = sample_dag(p, 2.0, "er", rng)
    X = simulate_sem(W_true, max(200, 20 * p), "gaussian", rng, standardize=False)
    Sigma = covariance(X)
    W = rng.normal(scale=0.15, size=(p, p))
    np.fill_diagonal(W, 0.0)
    mask = _mask(p)
    alpha = float(rng.uniform(0.0, 1.0))
    rho = float(rng.choice([1.0, 10.0, 100.0]))
    gamma = float(0.05 / max(np.linalg.norm(Sigma, 2), 1e-6))
    return Sigma, W, mask, alpha, rho, gamma, 0.1, 1.0


def load_sachs():
    """First 853 rows are the anti-CD3/CD28 condition in the standard table."""
    path = DATA / "cyto_full_data.csv"
    table = np.genfromtxt(path, delimiter=",", skip_header=1)
    X = table[:853]
    X = X - X.mean(axis=0, keepdims=True)
    truth = np.zeros((11, 11))
    for i, j in SACHS_EDGES:
        truth[i, j] = 1.0
    return X, truth


def run_sachs():
    X, truth = load_sachs()
    Sigma = covariance(X)
    trace = []
    t0 = time.perf_counter()
    solved = proximal_al(Sigma, lam=LAM, kind="poly", trace=trace)
    prox_seconds = time.perf_counter() - t0
    W_ref = solved["W"]
    W_hat = replay(Sigma, trace, len(trace), torch.float64)
    W_hat_one = replay(Sigma, trace, 1, torch.float64)
    methods = {"poly_prox": W_ref, "transformer": W_hat}
    timings = {"poly_prox": prox_seconds, "transformer": 0.0}
    runners = {
        "poly_lbfgs": lambda: notears_poly_lbfgs(Sigma, lam=LAM)["W"],
        "notears": lambda: notears_expm(Sigma, lam=LAM)["W"],
        "dagma": lambda: dagma_estimate(X, threshold=THRESHOLD),
        "golem": lambda: golem_ev(X, num_iter=GOLEM_STEPS, seed=0),
        "sort": lambda: sortnregress(X),
        "pc": lambda: pc_estimate(X),
        "ges": lambda: ges_estimate(X),
    }
    for name, fn in runners.items():
        t0 = time.perf_counter()
        methods[name] = np.asarray(fn(), dtype=np.float64)
        timings[name] = time.perf_counter() - t0
        print(f"sachs {name} shd={shd(truth, methods[name], THRESHOLD)} sec={timings[name]:.1f}", flush=True)
    summary = {}
    for name, W in methods.items():
        summary[name] = graph_metrics(truth, W)
        summary[name]["seconds"] = float(timings[name])
    summary["transformer"]["one_step"] = _norms(W_hat_one, trace[0]["W_out"])
    summary["transformer"]["full"] = _norms(W_hat, W_ref)
    summary["transformer"]["n_steps"] = len(trace)
    summary["poly_prox"]["h"] = float(solved["h"])
    np.savez(
        OUT / "sachs_graphs.npz",
        truth=truth,
        poly_prox=W_ref,
        transformer=W_hat,
        notears=methods["notears"],
        dagma=methods["dagma"],
        golem=methods["golem"],
    )
    print("sachs transformer full", summary["transformer"]["full"], flush=True)
    return summary


def run_baselines():
    rows = []
    specs = []
    for graph in ("er", "sf"):
        for noise in ("gaussian", "laplace"):
            for scale in ("raw", "standardized"):
                for p in (5, 10, 20):
                    for seed in (0, 1, 2):
                        specs.append((graph, noise, scale, p, seed))
    methods = (
        ("poly_prox", lambda Sigma, X, seed: proximal_al(Sigma, lam=LAM, kind="poly")["W"]),
        ("poly_lbfgs", lambda Sigma, X, seed: notears_poly_lbfgs(Sigma, lam=LAM)["W"]),
        ("notears", lambda Sigma, X, seed: notears_expm(Sigma, lam=LAM)["W"]),
        ("dagma", lambda Sigma, X, seed: dagma_estimate(X, threshold=THRESHOLD)),
        ("golem", lambda Sigma, X, seed: golem_ev(X, num_iter=GOLEM_STEPS, seed=seed)),
        ("sort", lambda Sigma, X, seed: sortnregress(X)),
        ("pc", lambda Sigma, X, seed: pc_estimate(X)),
        ("ges", lambda Sigma, X, seed: ges_estimate(X)),
    )
    for i, (graph, noise, scale, p, seed) in enumerate(specs, start=1):
        true_W, X, Sigma = _instance(p, 1000, seed, graph=graph, noise=noise, scale=scale)
        for name, fn in methods:
            t0 = time.perf_counter()
            try:
                est = np.asarray(fn(Sigma, X, seed), dtype=np.float64)
                metrics = graph_metrics(true_W, est)
                err = ""
            except Exception as exc:  # noqa: BLE001 — record and continue the grid
                metrics = {"shd": None, "precision": None, "recall": None, "f1": None, "acyclic": None, "sid": None}
                err = str(exc)
                est = None
            row = {
                "graph": graph,
                "noise": noise,
                "scale": scale,
                "p": p,
                "m": 1000,
                "degree": 2,
                "seed": seed,
                "method": name,
                "seconds": float(time.perf_counter() - t0),
                "error": err,
                **metrics,
            }
            rows.append(row)
            print(
                f"base {i}/{len(specs)} {name} {graph} {noise} {scale} p={p} seed={seed} "
                f"shd={row['shd']} sec={row['seconds']:.1f}",
                flush=True,
            )
    return rows


def main():
    path = OUT / "additional.json"
    payload = {}
    if path.exists():
        payload = json.loads(path.read_text())
    if "decomposition" not in payload:
        payload["decomposition"] = run_decomposition_and_raw()
        path.write_text(json.dumps(payload))
        print("wrote decomposition", flush=True)
    if "scaling" not in payload:
        payload["scaling"] = run_scaling()
        path.write_text(json.dumps(payload))
        print("wrote scaling", flush=True)
    if "sachs" not in payload:
        payload["sachs"] = run_sachs()
        path.write_text(json.dumps(payload))
        print("wrote sachs", flush=True)
    if "baselines" not in payload:
        payload["baselines"] = run_baselines()
        path.write_text(json.dumps(payload))
        print("wrote baselines", flush=True)
    print(path)


if __name__ == "__main__":
    main()
