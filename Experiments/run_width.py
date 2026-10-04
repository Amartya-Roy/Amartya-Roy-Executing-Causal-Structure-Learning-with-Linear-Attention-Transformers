"""Capacity-matched learnability check for Table 7.

Theorem 4.2 needs embedding width 13*p_max + 18 = 148 at p_max = 10; the models
of Table 5 are width 64. This repeats that experiment at width 160, which clears
the bound, with the data, loss, optimizer, epochs and evaluation unchanged, so
that the only difference is capacity.

Run:  python -m experiments.run_width            # paper-size corpus
      python -m experiments.run_width --large    # ~10x corpus, 2.5x epochs
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.run_train import (P_MAX, TiedAttention, build_dataset,
                                   one_step_error, train_model, unroll_eval)

OUT = Path(__file__).resolve().parent / "results"


def main(large=False):
    dev = torch.device("cpu")
    t0 = time.time()
    n_random, n_solver, epochs = (15000, 800, 30) if large else (1500, 80, 12)
    train = build_dataset(n_random, n_solver, p=5, seed=0)
    test5 = build_dataset(200, 0, p=5, seed=1)
    test10 = build_dataset(120, 0, p=10, seed=2)
    print(f"train {len(train)} test5 {len(test5)} test10 {len(test10)}", flush=True)

    report = {}
    for kind in ("softmax", "linear"):
        for d in (64, 160):
            rows = []
            for seed in range(5):
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = TiedAttention(P_MAX, d=d, kind=kind, use_memory=True)
                n_par = sum(q.numel() for q in model.parameters())
                hist = train_model(model, train, dev, epochs=epochs)
                med5, _ = one_step_error(model, test5, dev)
                med10, _ = one_step_error(model, test10, dev)
                traj = unroll_eval(model, dev, p=5, n_problems=12, seed=3)[-1]["dist_exact"]
                rows.append(dict(seed=seed, params=n_par, loss0=hist[0], lossT=hist[-1],
                                 p5=med5, p10=med10, traj12=traj))
                print(f"{kind} d{d} seed{seed}: p5 {med5:.3f} p10 {med10:.3f} "
                      f"traj {traj:.3f}  [{time.time() - t0:.0f}s]", flush=True)

            def mean_se(key):
                v = np.array([r[key] for r in rows], float)
                v = v[np.isfinite(v)]
                if v.size == 0:
                    return [float("nan"), float("nan"), 0]
                return [float(v.mean()), float(v.std(ddof=1) / max(v.size ** 0.5, 1)), int(v.size)]

            report[f"{kind}_d{d}"] = dict(
                kind=kind, d=d, params=rows[0]["params"], seeds=rows,
                p5=mean_se("p5"), p10=mean_se("p10"), traj12=mean_se("traj12"))

    path = OUT / ("width_match_large.json" if large else "width_match.json")
    path.write_text(json.dumps(report, indent=2))
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main(large="--large" in sys.argv)
