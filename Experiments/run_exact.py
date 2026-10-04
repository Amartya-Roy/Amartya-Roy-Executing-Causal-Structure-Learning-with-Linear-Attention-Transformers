"""Experiment 2: exact execution, multiplier growth, relative accuracy.

The block is the repeated-squaring update, one width p_max, applied at every
smaller p. Residuals are against an independent matrix-power reference in
float64. The real-arithmetic prediction is a zero residual; float32 and
bfloat16 are there to show the rounding that prediction ignores.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments.realization import masked_stationarity_residual, one_update, one_update_reference, pad_problem
from experiments.sem import covariance, sample_dag, simulate_sem

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def _as_dtype(arr, dtype):
    return torch.tensor(np.asarray(arr, dtype=np.float64), dtype=dtype)


def one_update_torch(Sigma, W, mask, alpha, rho, gamma, lam, b, N, dtype):
    Sigma_t = _as_dtype(Sigma, dtype)
    W_t = _as_dtype(W, dtype)
    mask_t = _as_dtype(mask, dtype)
    p = W_t.shape[0]
    eye = torch.eye(p, dtype=dtype)
    n_steps = int(np.log2(int(N)))
    n_t = torch.tensor(float(N), dtype=dtype)
    alpha_t = torch.tensor(float(alpha), dtype=dtype)
    rho_t = torch.tensor(float(rho), dtype=dtype)
    gamma_t = torch.tensor(float(gamma), dtype=dtype)
    lam_t = torch.tensor(float(lam), dtype=dtype)
    b_t = torch.tensor(float(b), dtype=dtype)

    def squaring(S, Q):
        for _ in range(n_steps):
            Q = Q + S + Q @ S
            S = 2 * S + S @ S
        return S, Q

    S, Q = squaring(W_t * W_t / n_t, torch.zeros_like(W_t))
    h = torch.trace(S)
    grad_h = 2 * W_t * Q.T
    grad = Sigma_t @ (W_t - eye) + (alpha_t + rho_t * h) * grad_h
    V = W_t - gamma_t * grad
    tau = gamma_t * lam_t
    W_next = mask_t * (torch.sign(V) * torch.clamp(V.abs() - tau, min=0))
    S2, _ = squaring(W_next * W_next / n_t, torch.zeros_like(W_t))
    h_next = torch.trace(S2)
    alpha_next = alpha_t + b_t * rho_t * h_next
    W_np = W_next.detach().to(torch.float64).cpu().numpy()
    return W_np, float(alpha_next.detach().to(torch.float64)), float(h.detach().to(torch.float64)), float(h_next.detach().to(torch.float64))


def random_state(p, rng):
    W_true = sample_dag(p, 2.0, "er", rng)
    X = simulate_sem(W_true, max(200, 20 * p), "gaussian", rng, standardize=False)
    Sigma = covariance(X)
    W = rng.normal(scale=0.15, size=(p, p))
    np.fill_diagonal(W, 0.0)
    mask = np.ones((p, p))
    np.fill_diagonal(mask, 0.0)
    alpha = float(rng.uniform(0.0, 1.0))
    rho = float(rng.choice([1.0, 10.0, 100.0]))
    gamma = float(0.05 / max(np.linalg.norm(Sigma, 2), 1e-6))
    lam = 0.1
    b = 1.0
    return Sigma, W, mask, alpha, rho, gamma, lam, b


def run_residuals(p_max=20, sizes=(3, 5, 10, 20), n_states=8, seed=0):
    rng = np.random.default_rng(seed)
    N = 1 << int(np.ceil(np.log2(p_max)))
    rows = []
    dtypes = {
        "float64": torch.float64,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    for p in sizes:
        for s in range(n_states):
            Sigma, W, mask, alpha, rho, gamma, lam, b = random_state(p, rng)
            Sigma_p, W_p, mask_p = pad_problem(Sigma, W, mask, p_max)
            ref_W, ref_a, ref_h, ref_hp = one_update_reference(
                Sigma, W, mask, alpha, rho, gamma, lam, b, N
            )
            for name, dt in dtypes.items():
                got_W, got_a, got_h, got_hp = one_update_torch(
                    Sigma_p, W_p, mask_p, alpha, rho, gamma, lam, b, N, dt
                )
                active = got_W[:p, :p]
                tail = got_W.copy()
                tail[:p, :p] = 0.0
                rows.append({
                    "p": p,
                    "state": s,
                    "dtype": name,
                    "W_residual": float(np.linalg.norm(active - ref_W)),
                    "alpha_residual": float(abs(got_a - ref_a)),
                    "h_residual": float(abs(got_h - ref_h)),
                    "pad_residual": float(np.linalg.norm(tail)),
                    "ref_h": ref_h,
                })
            # Independent check: numpy squaring versus matrix_power, same dtype.
            sq_W, sq_a, _, _ = one_update(Sigma, W, mask, alpha, rho, gamma, lam, b, N)
            rows.append({
                "p": p,
                "state": s,
                "dtype": "squaring_vs_power",
                "W_residual": float(np.linalg.norm(sq_W - ref_W)),
                "alpha_residual": float(abs(sq_a - ref_a)),
                "h_residual": float("nan"),
                "pad_residual": 0.0,
                "ref_h": ref_h,
            })
    return rows


def explicit_multiplier(r=0.6, lam=0.2):
    """The two-node example under Proposition (quantitative multiplier growth)."""
    Sigma = np.array([[1.0, r], [r, 1.0]])
    rows = []
    for t in np.logspace(-2, -0.4, 12):
        W = np.array([[0.0, t], [t, 0.0]])
        N = 2
        from experiments.hgrad import h_and_grad

        h, gh = h_and_grad(W, N)
        c = (r - lam - t) / (t ** 3)
        dist = float(np.linalg.norm(W))
        d_M, gL, mnorm = masked_stationarity_residual(Sigma, W, lam)
        # At the explicit stationary choice of c, the augmented residual is zero
        # on the allowed entries, so eta = 0 and the sandwich uses d_M and C_k.
        C_k = gL + lam * mnorm
        rows.append({
            "t": float(t),
            "h": h,
            "grad_entry": float(gh[0, 1]),
            "grad_entry_formula": float(t ** 3),
            "h_formula": float(t ** 4 / 2),
            "c": float(c),
            "c_times_grad_norm": float(c * np.linalg.norm(gh)),
            "d_M": d_M,
            "C_k": C_k,
            "dist": dist,
            "c_asymptotic": float((r - lam) / t ** 3),
        })
    return rows


def relative_accuracy(nu_grid=(1e-3, 1e-2, 5e-2, 1e-1)):
    """Corollary check on the explicit stationary family, where eta = 0.

    A relative error in h and grad h stays inside (2 nu + nu^2) C_k.
    A fixed absolute perturbation of grad h is multiplied by c.
    """
    r, lam = 0.6, 0.2
    Sigma = np.array([[1.0, r], [r, 1.0]])
    from experiments.hgrad import h_and_grad

    rows = []
    for t in (0.2, 0.1, 0.05, 0.02):
        W = np.array([[0.0, t], [t, 0.0]])
        h, gh = h_and_grad(W, 2)
        c = (r - lam - t) / (t ** 3)
        alpha = c  # rho = 0 in this stationary example, so c = alpha
        rho = 0.0
        g = Sigma @ (W - np.eye(2)) + c * gh
        _d, gL, mnorm = masked_stationarity_residual(Sigma, W, lam)
        C_k = gL + lam * mnorm
        for nu in nu_grid:
            h_rel = h * (1.0 + nu)
            H_rel = gh * (1.0 + nu)
            c_rel = alpha + rho * h_rel
            g_rel = Sigma @ (W - np.eye(2)) + c_rel * H_rel
            rel_err = float(np.linalg.norm(g_rel - g))
            # Fixed absolute perturbation, independent of the scale of grad h.
            eps_abs = 1e-4
            delta = np.array([[0.0, eps_abs], [eps_abs, 0.0]])
            g_abs = Sigma @ (W - np.eye(2)) + c * (gh + delta)
            abs_err = float(np.linalg.norm(g_abs - g))
            rows.append({
                "t": float(t),
                "c": float(c),
                "nu": float(nu),
                "relative_grad_error": rel_err,
                "bound": float((2 * nu + nu ** 2) * C_k),
                "absolute_grad_error": abs_err,
                "absolute_eps": eps_abs,
                "C_k": float(C_k),
                "within_bound": bool(rel_err <= (2 * nu + nu ** 2) * C_k * (1 + 1e-8) + 1e-10),
            })
    return rows


def permutation_check(n=6, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    N = 32
    for p in (3, 5, 10):
        Sigma, W, mask, alpha, rho, gamma, lam, b = random_state(p, rng)
        perm = rng.permutation(p)
        P = np.eye(p)[perm]
        W1, a1, _, _ = one_update(Sigma, W, mask, alpha, rho, gamma, lam, b, N)
        Sigma2 = P.T @ Sigma @ P
        W2 = P.T @ W @ P
        mask2 = P.T @ mask @ P
        Wp, ap, _, _ = one_update(Sigma2, W2, mask2, alpha, rho, gamma, lam, b, N)
        conjugated = P.T @ W1 @ P
        rows.append({
            "p": p,
            "W_residual": float(np.linalg.norm(Wp - conjugated)),
            "alpha_residual": float(abs(ap - a1)),
        })
    return rows


def alpha_drift(steps=20, eps=1e-3):
    """Remark: near an acyclic state, an error in alpha is not damped."""
    rng = np.random.default_rng(0)
    p = 4
    W_true = sample_dag(p, 2.0, "er", rng)
    X = simulate_sem(W_true, 400, "gaussian", rng, False)
    Sigma = covariance(X)
    lam = float(np.max(np.abs(Sigma)) + 0.1)
    mask = np.ones((p, p))
    np.fill_diagonal(mask, 0.0)
    gamma = 0.1 / max(np.linalg.norm(Sigma, 2), 1e-6)
    W = np.zeros((p, p))
    alpha = 0.0
    history = []
    for ell in range(steps):
        # Exact step stays at 0 because lambda dominates every covariance entry.
        W_exact, alpha_exact, _, _ = one_update(Sigma, W, mask, alpha, 1.0, gamma, lam, 1.0, 8)
        alpha_hat = alpha + eps
        W_hat, alpha_hat_next, _, _ = one_update(Sigma, W, mask, alpha_hat, 1.0, gamma, lam, 0.0, 8)
        history.append({
            "ell": ell,
            "exact_W": float(np.linalg.norm(W_exact)),
            "hat_alpha": float(alpha_hat),
            "hat_W": float(np.linalg.norm(W_hat)),
            "predicted_alpha_error": float((ell + 1) * eps),
        })
        W = W_exact
        alpha = alpha_exact
        # The drifting decoder keeps the inflated multiplier.
        alpha = alpha_hat
    return {"lambda": lam, "eps": eps, "history": history}


def main():
    residual_rows = run_residuals()
    summary = []
    for dtype in ("squaring_vs_power", "float64", "float32", "bfloat16"):
        for p in (3, 5, 10, 20):
            chunk = [r for r in residual_rows if r["dtype"] == dtype and r["p"] == p]
            vals = np.array([r["W_residual"] for r in chunk])
            summary.append({
                "dtype": dtype,
                "p": p,
                "median_W": float(np.median(vals)),
                "max_W": float(np.max(vals)),
                "max_alpha": float(np.max([r["alpha_residual"] for r in chunk])),
                "max_pad": float(np.max([r["pad_residual"] for r in chunk])),
            })
    payload = {
        "residuals": residual_rows,
        "summary": summary,
        "multiplier": explicit_multiplier(),
        "relative_accuracy": relative_accuracy(),
        "permutation": permutation_check(),
        "alpha_drift": alpha_drift(),
    }
    path = OUT / "exact_execution.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}")
    for row in summary:
        print(
            f"{row['dtype']:20s} p={row['p']:2d}  "
            f"median {row['median_W']:.3e}  max {row['max_W']:.3e}  "
            f"pad {row['max_pad']:.3e}"
        )
    rel = payload["relative_accuracy"]
    print("relative bound holds:", all(r["within_bound"] for r in rel))
    perm = payload["permutation"]
    print("permutation max W", max(r["W_residual"] for r in perm))


if __name__ == "__main__":
    main()
