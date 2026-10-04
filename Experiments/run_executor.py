"""Verification that the constructed network executes one Poly-prox update.

Three checks, all on the *actual* attention/feedforward forward pass of
experiments/executor.py, whose weights are fixed analytically from p_max alone:

  1. equivalence   -- the forward pass against an independent matrix-power
                      implementation of Algorithm 1, over random masks, problem
                      sizes, controls and states;
  2. bridge        -- the forward pass against experiments/realization.py, the
                      direct arithmetic implementation used for the large-scale
                      recovery runs, which licenses using the faster one there;
  3. cost          -- realized depth, heads per layer, key dimension, width,
                      unit counts and weight magnitudes, against the values
                      claimed in the realization theorem.

Run:  python -m experiments.run_executor
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.executor import build, cost, run
from experiments.realization import one_update, one_update_reference

OUT = Path(__file__).resolve().parent / "results" / "executor.json"


def random_case(rng, p):
    A = rng.normal(size=(p, p))
    Sigma = A @ A.T / max(p, 1)
    M = (rng.random((p, p)) < rng.choice([1.0, 0.6, 0.25])).astype(float)
    np.fill_diagonal(M, 0.0)
    W = rng.normal(size=(p, p)) * rng.choice([0.05, 0.3, 0.8]) * M
    return dict(
        Sigma=Sigma, M=M, W=W,
        alpha=float(rng.normal() * rng.choice([0.0, 1.0, 50.0])),
        lam=float(abs(rng.normal()) * rng.choice([0.0, 1.0, 10.0])),
        rho=float(abs(rng.normal()) * rng.choice([0.0, 1.0, 20.0])),
        gamma=float(abs(rng.normal()) * rng.choice([0.0, 0.1, 1.0])),
        b=float(rng.integers(0, 2)),
    )


def equivalence(p_maxes=(4, 8, 16, 32), trials=15, seed=11):
    rng = np.random.default_rng(seed)
    worst_W = worst_a = 0.0
    n = skipped = 0
    for p_max in p_maxes:
        layers, lay, N, _ = build(p_max)
        for p in sorted({1, 2, max(1, p_max // 4), max(1, p_max // 2), p_max}):
            for _ in range(trials):
                c = random_case(rng, p)
                Wr, ar, _, _ = one_update_reference(
                    c["Sigma"], c["W"], c["M"], c["alpha"], c["rho"],
                    c["gamma"], c["lam"], c["b"], N)
                if not (np.isfinite(Wr).all() and np.isfinite(ar)):
                    skipped += 1          # the polynomial constraint overflowed
                    continue
                Wt, at = run(c["Sigma"], c["M"], c["W"], c["alpha"], c["lam"],
                             c["rho"], c["gamma"], c["b"], p_max, layers, lay)
                worst_W = max(worst_W, np.abs(Wt - Wr).max() / max(1.0, np.abs(Wr).max()))
                worst_a = max(worst_a, abs(at - ar) / max(1.0, abs(ar)))
                n += 1
    return {"cases": n, "skipped_overflow": skipped,
            "worst_rel_W": worst_W, "worst_rel_alpha": worst_a}


def bridge(p_maxes=(8, 16, 32), trials=20, seed=23):
    """Forward pass vs the direct arithmetic implementation used at scale."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    n = 0
    for p_max in p_maxes:
        layers, lay, N, _ = build(p_max)
        for p in sorted({2, max(1, p_max // 2), p_max}):
            for _ in range(trials):
                c = random_case(rng, p)
                Wa, aa, _, _ = one_update(c["Sigma"], c["W"], c["M"], c["alpha"],
                                          c["rho"], c["gamma"], c["lam"], c["b"], N)
                if not np.isfinite(Wa).all():
                    continue
                Wt, at = run(c["Sigma"], c["M"], c["W"], c["alpha"], c["lam"],
                             c["rho"], c["gamma"], c["b"], p_max, layers, lay)
                worst = max(worst, np.abs(Wt - Wa).max() / max(1.0, np.abs(Wa).max()))
                n += 1
    return {"cases": n, "worst_rel_W": worst}


def invariances(p_max=16, seed=5):
    rng = np.random.default_rng(seed)
    layers, lay, N, _ = build(p_max)
    p = 6
    A = rng.normal(size=(p, p))
    Sigma, M = A @ A.T / p, 1.0 - np.eye(p)
    W = rng.normal(size=(p, p)) * 0.3 * M
    args = dict(alpha=0.5, lam=0.1, rho=1.0, gamma=0.05, b=1.0)
    W1, _ = run(Sigma, M, W, args["alpha"], args["lam"], args["rho"], args["gamma"], args["b"],
                p_max, layers, lay)
    P = np.eye(p)[rng.permutation(p)]
    W2, _ = run(P.T @ Sigma @ P, P.T @ M @ P, P.T @ W @ P, args["alpha"], args["lam"],
                args["rho"], args["gamma"], args["b"], p_max, layers, lay)
    perm = float(np.abs(P.T @ W1 @ P - W2).max())
    pads = {}
    for pm in (8, 16, 32):
        L2, l2, N2, _ = build(pm)
        Wt, _ = run(Sigma, M, W, args["alpha"], args["lam"], args["rho"], args["gamma"],
                    args["b"], pm, L2, l2)
        Wr, _, _, _ = one_update_reference(Sigma, W, M, args["alpha"], args["rho"],
                                           args["gamma"], args["lam"], args["b"], N2)
        pads[str(pm)] = float(np.abs(Wt - Wr).max())
    return {"permutation_equivariance": perm, "padding_same_p_various_pmax": pads}


def timing(p_max=20, reps=20, seed=3):
    rng = np.random.default_rng(seed)
    layers, lay, _, _ = build(p_max)
    c = random_case(rng, p_max)
    run(c["Sigma"], c["M"], c["W"], c["alpha"], c["lam"], c["rho"], c["gamma"], c["b"],
        p_max, layers, lay)
    t0 = time.perf_counter()
    for _ in range(reps):
        run(c["Sigma"], c["M"], c["W"], c["alpha"], c["lam"], c["rho"], c["gamma"], c["b"],
            p_max, layers, lay)
    return {"p_max": p_max, "seconds_per_forward_pass": (time.perf_counter() - t0) / reps}


def main():
    res = {
        "equivalence": equivalence(),
        "bridge_to_arithmetic_implementation": bridge(),
        "invariances": invariances(),
        "cost": [cost(pm) for pm in (8, 16, 20, 32, 64, 128)],
        "timing": timing(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["equivalence"], indent=2))
    print(json.dumps(res["bridge_to_arithmetic_implementation"], indent=2))
    print(json.dumps(res["invariances"], indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
