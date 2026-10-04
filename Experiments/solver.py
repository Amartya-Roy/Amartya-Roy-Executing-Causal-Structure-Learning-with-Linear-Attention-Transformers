"""Proximal augmented-Lagrangian solver for the polynomial constraint.

This is the recurrence (w update, alpha update): several masked
soft-threshold steps with the multiplier held fixed, then one multiplier
step. A second driver uses L-BFGS on the same augmented Lagrangian so the
constraint and the optimizer can be separated.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from experiments.hgrad import h_and_grad, h_and_grad_expm


def soft_threshold(V: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:
        return V
    return np.sign(V) * np.maximum(np.abs(V) - tau, 0.0)


def least_squares(Sigma: np.ndarray, W: np.ndarray) -> float:
    p = W.shape[0]
    R = np.eye(p) - W
    return 0.5 * float(np.trace(R.T @ Sigma @ R))


def score(Sigma: np.ndarray, W: np.ndarray, lam: float) -> float:
    return least_squares(Sigma, W) + lam * float(np.sum(np.abs(W)))


def _smooth(Sigma: np.ndarray, W: np.ndarray, alpha: float, rho: float, N: int, kind: str):
    if kind == "poly":
        h, gh = h_and_grad(W, N)
    elif kind == "expm":
        h, gh = h_and_grad_expm(W)
    else:
        raise ValueError(kind)
    p = W.shape[0]
    loss = least_squares(Sigma, W)
    val = loss + alpha * h + 0.5 * rho * h * h
    grad = Sigma @ (W - np.eye(p)) + (alpha + rho * h) * gh
    return val, grad, h, gh


def prox_map(W, grad, gamma, lam, mask):
    V = W - gamma * grad
    return mask * soft_threshold(V, gamma * lam)


def proximal_al(
    Sigma: np.ndarray,
    lam: float = 0.1,
    kind: str = "poly",
    max_outer: int = 12,
    max_inner: int = 400,
    h_tol: float = 1e-8,
    stat_tol: float = 1e-6,
    rho_max: float = 1e16,
    trace=None,
    alpha_log=None,
    step_fn=None,
    alpha_fn=None,
):
    """Run the paper's proximal recurrence. `kind` is 'poly' or 'expm'.

    `step_fn(W, alpha, rho, gamma) -> W_new` and `alpha_fn(W, alpha, rho) ->
    alpha_new`, when given, replace the proximal step and the multiplier update
    by an external executor (for example the fixed-weight transformer block).
    The controller -- line search, penalty schedule, stopping rule -- stays
    here, as in Corollary "exact finite-horizon execution".
    """
    p = Sigma.shape[0]
    N = 1 if p <= 1 else 1 << int(np.ceil(np.log2(p)))
    mask = np.ones((p, p))
    np.fill_diagonal(mask, 0.0)
    W = np.zeros((p, p))
    alpha = 0.0
    rho = 1.0
    h_prev = np.inf
    eye_norm = float(np.linalg.norm(Sigma, 2))
    gamma = 1.0 / max(eye_norm, 1e-6)
    last_h = np.inf
    last_grad_norm = np.inf
    n_accepted = 0

    for _outer in range(max_outer):
        for _inner in range(max_inner):
            val, grad, h, _gh = _smooth(Sigma, W, alpha, rho, N, kind)
            if not np.isfinite(val):
                break
            accepted = False
            gtry = gamma
            W_new = W
            for _bt in range(25):
                if step_fn is None:
                    W_new = prox_map(W, grad, gtry, lam, mask)
                else:
                    W_new = step_fn(W, alpha, rho, gtry)
                dW = W_new - W
                val_new, _, _, _ = _smooth(Sigma, W_new, alpha, rho, N, kind)
                if not np.isfinite(val_new):
                    gtry *= 0.5
                    continue
                rhs = val + float(np.sum(grad * dW)) + (0.5 / gtry) * float(np.sum(dW * dW))
                if val_new <= rhs + 1e-8 * (1.0 + abs(val)):
                    accepted = True
                    break
                gtry *= 0.5
            if not accepted:
                break
            if trace is not None:
                # Controls of this accepted proximal step. The multiplier is
                # held fixed inside the inner loop, so b = 0 here.
                trace.append({
                    "alpha": float(alpha),
                    "rho": float(rho),
                    "gamma": float(gtry),
                    "W_out": W_new.copy(),
                })
            n_accepted += 1
            gap = float(np.linalg.norm(W_new - W)) / max(gtry, 1e-16)
            W = W_new
            gamma = min(gtry * 1.2, 10.0 / max(eye_norm, 1e-6))
            last_h = h
            last_grad_norm = gap
            if gap < stat_tol:
                break
        _val, grad, h, _gh = _smooth(Sigma, W, alpha, rho, N, kind)
        last_h = h
        # Proximal-gradient residual at the current step length.
        W_fixed = prox_map(W, grad, gamma, lam, mask)
        last_grad_norm = float(np.linalg.norm(W - W_fixed)) / max(gamma, 1e-16)
        if h <= h_tol or rho >= rho_max:
            alpha = alpha + rho * h if alpha_fn is None else alpha_fn(W, alpha, rho)
            if alpha_log is not None:
                alpha_log.append({
                    "rho": float(rho),
                    "h": float(h),
                    "alpha": float(alpha),
                    "n_steps": int(n_accepted),
                })
            break
        if h > 0.25 * h_prev:
            rho = min(rho * 10.0, rho_max)
        alpha = alpha + rho * h if alpha_fn is None else alpha_fn(W, alpha, rho)
        if alpha_log is not None:
            alpha_log.append({
                "rho": float(rho),
                "h": float(h),
                "alpha": float(alpha),
                "n_steps": int(n_accepted),
            })
        h_prev = h

    return {
        "W": W,
        "alpha": alpha,
        "rho": rho,
        "h": float(last_h),
        "score": score(Sigma, W, lam),
        "G_gamma": float(last_grad_norm),
        "gamma": float(gamma),
    }


def lbfgs_al(
    Sigma: np.ndarray,
    lam: float = 0.1,
    kind: str = "expm",
    max_outer: int = 40,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
):
    """NOTEARS-style L-BFGS augmented Lagrangian. `kind` is 'expm' or 'poly'."""
    p = Sigma.shape[0]
    N = 1 if p <= 1 else 1 << int(np.ceil(np.log2(p)))
    rho = 1.0
    alpha = 0.0
    h_val = np.inf
    w0 = np.zeros(p * p)

    def objective(w):
        W = w.reshape(p, p).copy()
        np.fill_diagonal(W, 0.0)
        if kind == "poly":
            h, gh = h_and_grad(W, N)
        else:
            h, gh = h_and_grad_expm(W)
        loss = least_squares(Sigma, W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lam * float(np.sum(np.abs(W)))
        grad = Sigma @ (W - np.eye(p)) + (rho * h + alpha) * gh + lam * np.sign(W)
        np.fill_diagonal(grad, 0.0)
        return obj, grad.reshape(-1)

    for _ in range(max_outer):
        while rho < rho_max:
            sol = minimize(
                objective,
                w0,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 200, "ftol": 1e-12},
            )
            w_new = sol.x
            W = w_new.reshape(p, p).copy()
            np.fill_diagonal(W, 0.0)
            if kind == "poly":
                h_new, _ = h_and_grad(W, N)
            else:
                h_new, _ = h_and_grad_expm(W)
            if h_new > 0.25 * h_val:
                rho *= 10.0
            else:
                break
        w0 = w_new
        h_val = float(h_new)
        alpha += rho * h_val
        if h_val <= h_tol or rho >= rho_max:
            break

    W = w0.reshape(p, p).copy()
    np.fill_diagonal(W, 0.0)
    # Stationarity residual of the smooth part plus the L1 subgradient, on free entries.
    if kind == "poly":
        h, gh = h_and_grad(W, N)
    else:
        h, gh = h_and_grad_expm(W)
    grad = Sigma @ (W - np.eye(p)) + (rho * h + alpha) * gh
    mask = np.ones((p, p))
    np.fill_diagonal(mask, 0.0)
    # Distance from -grad to the L1 subdifferential, entrywise.
    residual = np.zeros_like(W)
    for i in range(p):
        for j in range(p):
            if mask[i, j] == 0:
                continue
            g = grad[i, j]
            w = W[i, j]
            if abs(w) > 1e-12:
                residual[i, j] = g + lam * np.sign(w)
            else:
                residual[i, j] = np.sign(g) * max(abs(g) - lam, 0.0)
    return {
        "W": W,
        "alpha": float(alpha),
        "rho": float(rho),
        "h": float(h),
        "score": score(Sigma, W, lam),
        "G_gamma": float(np.linalg.norm(residual)),
        "gamma": float("nan"),
    }
