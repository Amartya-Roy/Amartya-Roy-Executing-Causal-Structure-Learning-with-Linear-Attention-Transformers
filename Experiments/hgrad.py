"""Acyclicity value and gradient, by repeated squaring and by matrix power."""

from __future__ import annotations

import numpy as np


def squaring_powers(W: np.ndarray, N: int):
    """Return S = Y^N - I and Q = Y^{N-1} - I, with Y = I + (W odot W) / N.

    The recurrence is the one in the realization proof: S <- 2S + S^2,
    Q <- Q + S + Q S, started from S = (W odot W) / N and Q = 0.
    """
    if N < 1 or (N & (N - 1)) != 0:
        raise ValueError("N must be a power of two")
    B = W * W
    S = B / N
    Q = np.zeros_like(W)
    steps = int(np.log2(N))
    for _ in range(steps):
        Q = Q + S + Q @ S
        S = 2.0 * S + S @ S
    return S, Q


def h_and_grad(W: np.ndarray, N: int):
    """Polynomial constraint h(W) and its exact gradient 2 W odot (Y^{N-1})^T."""
    S, Q = squaring_powers(W, N)
    h = float(np.trace(S))
    grad = 2.0 * W * Q.T
    return h, grad


def h_and_grad_power(W: np.ndarray, N: int):
    """Same quantities via np.linalg.matrix_power, for an independent check."""
    p = W.shape[0]
    Y = np.eye(p) + (W * W) / N
    if N == 1:
        YN = Y
        YNm1 = np.eye(p)
    else:
        YN = np.linalg.matrix_power(Y, N)
        YNm1 = np.linalg.matrix_power(Y, N - 1)
    h = float(np.trace(YN) - p)
    grad = 2.0 * W * YNm1.T
    return h, grad


def h_and_grad_expm(W: np.ndarray):
    """Matrix-exponential constraint tr(exp(W odot W)) - p and its gradient."""
    from scipy.linalg import expm

    p = W.shape[0]
    E = expm(W * W)
    h = float(np.trace(E) - p)
    grad = 2.0 * W * E.T
    return h, grad
