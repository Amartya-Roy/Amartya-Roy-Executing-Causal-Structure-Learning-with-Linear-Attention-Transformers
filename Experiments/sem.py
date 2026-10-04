"""Linear SEM simulations used by the planned evaluation.

Edge convention matches the paper: W[i, j] is the weight of i -> j, and
each sample satisfies x_j = sum_i W[i, j] x_i + noise_j.
Expected degree k means each node has k neighbors in expectation
(k * p / 2 directed edges).
"""

from __future__ import annotations

import numpy as np


def polynomial_degree(p: int) -> int:
    """Smallest N = 2^s with N >= p, as in eq. (h)."""
    if p <= 1:
        return 1
    s = int(np.ceil(np.log2(p)))
    return 1 << s


def _orient_by_order(undirected_edges, order):
    """Orient each undirected edge from the earlier node in `order` to the later one."""
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    directed = []
    for i, j in undirected_edges:
        if rank[i] < rank[j]:
            directed.append((i, j))
        else:
            directed.append((j, i))
    return directed


def er_edges(p: int, expected_degree: float, rng: np.random.Generator):
    """Forward edges in a random topological order, each with probability k/(p-1)."""
    order = rng.permutation(p)
    rank = np.empty(p, dtype=int)
    rank[order] = np.arange(p)
    if p == 1:
        return []
    prob = float(expected_degree) / (p - 1)
    edges = []
    for a in range(p):
        for b in range(a + 1, p):
            i, j = int(order[a]), int(order[b])
            if rng.random() < prob:
                edges.append((i, j))
    return edges


def sf_edges(p: int, expected_degree: float, rng: np.random.Generator):
    """Preferential-attachment DAG. Attachments are parents in a growing order,
    then the node labels are randomly permuted."""
    if p == 1:
        return []
    m = max(1, int(round(expected_degree / 2.0)))
    degree = np.zeros(p, dtype=float)
    raw = []
    for new in range(1, p):
        n_attach = min(m, new)
        weights = degree[:new] + 1.0
        weights = weights / weights.sum()
        parents = rng.choice(new, size=n_attach, replace=False, p=weights)
        for par in parents:
            raw.append((int(par), new))
            degree[par] += 1.0
            degree[new] += 1.0
    if expected_degree < 1.5 * m:
        # Thin so that degree 1 is not forced up to the BA minimum of about 2.
        keep_prob = float(expected_degree) / max(2.0 * m, 1e-8)
        keep_prob = min(1.0, keep_prob)
        raw = [e for e in raw if rng.random() < keep_prob]
    perm = rng.permutation(p)
    return [(int(perm[i]), int(perm[j])) for i, j in raw]


def sample_dag(p: int, expected_degree: float, graph: str, rng: np.random.Generator,
                weight_scale=(0.5, 2.0)):
    """Return a weighted acyclic adjacency matrix with the paper's edge convention."""
    if graph == "er":
        edges = er_edges(p, expected_degree, rng)
    elif graph == "sf":
        edges = sf_edges(p, expected_degree, rng)
    else:
        raise ValueError(graph)
    W = np.zeros((p, p), dtype=float)
    lo, hi = weight_scale
    for i, j in edges:
        mag = rng.uniform(lo, hi)
        sign = rng.choice([-1.0, 1.0])
        W[i, j] = sign * mag
    return W


def sample_noise(m: int, p: int, kind: str, rng: np.random.Generator):
    if kind == "gaussian":
        return rng.normal(size=(m, p))
    if kind == "laplace":
        # Unit variance: Laplace(0, b) has variance 2 b^2.
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=(m, p))
    raise ValueError(kind)


def simulate_sem(W: np.ndarray, m: int, noise: str, rng: np.random.Generator,
                  standardize: bool):
    """Draw m centered samples. `standardize=True` also rescales each column to unit variance."""
    p = W.shape[0]
    E = sample_noise(m, p, noise, rng)
    # X = X W + E  =>  X (I - W) = E.
    A = np.eye(p) - W
    X = np.linalg.solve(A.T, E.T).T
    X = X - X.mean(axis=0, keepdims=True)
    if standardize:
        std = X.std(axis=0, ddof=0)
        std[std < 1e-12] = 1.0
        X = X / std
    return X


def covariance(X: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    return (X.T @ X) / m
