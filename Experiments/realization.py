"""One proximal-gradient-plus-multiplier step, in a chosen floating-point dtype.

The arithmetic is the layer schedule of the realization: two repeated-squaring
passes around a masked soft-threshold. Padding to p_max uses one width, as in
the theorem that a single weight set serves every p <= p_max.
"""

from __future__ import annotations

import numpy as np


def _soft(V, tau):
    return np.sign(V) * np.maximum(np.abs(V) - tau, 0.0)


def one_update(Sigma, W, mask, alpha, rho, gamma, lam, b, N):
    """Exact arithmetic of (w update, alpha update), in whatever dtype the arrays use."""
    p = W.shape[0]
    eye = np.eye(p, dtype=W.dtype)
    steps = int(np.log2(int(N)))

    def squaring(S, Q):
        for _ in range(steps):
            Q = Q + S + Q @ S
            S = 2.0 * S + S @ S
        return S, Q

    B = W * W
    S = B / np.array(N, dtype=W.dtype)
    Q = np.zeros_like(W)
    S, Q = squaring(S, Q)
    h = np.trace(S).real if np.iscomplexobj(S) else np.trace(S)
    grad_h = 2.0 * W * Q.T
    grad = Sigma @ (W - eye) + (alpha + rho * h) * grad_h
    V = W - gamma * grad
    W_next = mask * _soft(V, gamma * lam)

    B2 = W_next * W_next
    S2 = B2 / np.array(N, dtype=W.dtype)
    Q2 = np.zeros_like(W)
    S2, _Q2 = squaring(S2, Q2)
    h_next = np.trace(S2)
    alpha_next = alpha + b * rho * h_next
    return W_next, alpha_next, h, h_next


def one_update_reference(Sigma, W, mask, alpha, rho, gamma, lam, b, N):
    """Float64 reference that computes Y^N by matrix_power rather than squaring."""
    Sigma = np.asarray(Sigma, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    p = W.shape[0]
    Y = np.eye(p) + (W * W) / N
    YN = np.linalg.matrix_power(Y, N)
    YNm1 = np.linalg.matrix_power(Y, N - 1) if N > 1 else np.eye(p)
    h = float(np.trace(YN) - p)
    grad_h = 2.0 * W * YNm1.T
    grad = Sigma @ (W - np.eye(p)) + (alpha + rho * h) * grad_h
    V = W - gamma * grad
    W_next = mask * _soft(V, gamma * lam)
    Yp = np.eye(p) + (W_next * W_next) / N
    h_next = float(np.trace(np.linalg.matrix_power(Yp, N)) - p)
    alpha_next = alpha + b * rho * h_next
    return W_next, alpha_next, h, h_next


def pad_problem(Sigma, W, mask, p_max):
    p = W.shape[0]
    def pad(A):
        out = np.zeros((p_max, p_max), dtype=A.dtype)
        out[:p, :p] = A
        return out
    return pad(Sigma), pad(W), pad(mask)


def masked_stationarity_residual(Sigma, W, lam):
    """d_M(W): distance from 0 to the masked Lasso subdifferential."""
    p = W.shape[0]
    grad_L = Sigma @ (W - np.eye(p))
    mask = np.ones((p, p))
    np.fill_diagonal(mask, 0.0)
    residual = np.zeros_like(W)
    for i in range(p):
        for j in range(p):
            if mask[i, j] == 0.0:
                continue
            g = grad_L[i, j]
            w = W[i, j]
            if abs(w) > 1e-14:
                residual[i, j] = g + lam * np.sign(w)
            else:
                residual[i, j] = np.sign(g) * max(abs(g) - lam, 0.0)
    return float(np.linalg.norm(residual)), float(np.linalg.norm(mask * grad_L)), float(np.linalg.norm(mask))
