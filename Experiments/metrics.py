"""Graph metrics: SHD, directed precision/recall, SID, varsortability, acyclicity."""

from __future__ import annotations

from collections import deque

import numpy as np


def binarize(W: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    A = (np.abs(W) > threshold).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A


def edge_type(A: np.ndarray, i: int, j: int) -> str:
    """Classify the unordered pair {i, j}. Both directions set means undirected."""
    fwd = bool(A[i, j])
    back = bool(A[j, i])
    if fwd and back:
        return "undirected"
    if fwd:
        return "forward"
    if back:
        return "backward"
    return "none"


def shd(true_W: np.ndarray, est_W: np.ndarray, threshold: float = 0.0) -> int:
    """Missing, extra, or reversed edges. A reversal counts once.

    An undirected estimate against a directed truth counts as one mismatch.
    """
    T = binarize(true_W, 0.0)
    E = binarize(est_W, threshold)
    p = T.shape[0]
    err = 0
    for i in range(p):
        for j in range(i + 1, p):
            a = edge_type(T, i, j)
            b = edge_type(E, i, j)
            if a != b:
                err += 1
    return err


def directed_precision_recall(true_W: np.ndarray, est_W: np.ndarray, threshold: float = 0.0):
    """Precision and recall on directed edges. An undirected mark is not a true positive."""
    T = binarize(true_W, 0.0)
    E = binarize(est_W, threshold)
    # Drop the undirected part of E from the positive set: a positive is a
    # one-way edge.
    undirected = (E == 1) & (E.T == 1)
    E = E.copy()
    E[undirected] = 0
    tp = int(np.sum((T == 1) & (E == 1)))
    fp = int(np.sum((T == 0) & (E == 1)))
    fn = int(np.sum((T == 1) & (E == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall


def is_acyclic(A: np.ndarray) -> bool:
    """A is a 0/1 matrix, edge i -> j. Undirected pairs (both directions) are cycles."""
    p = A.shape[0]
    indeg = A.sum(axis=0).astype(int)
    children = [np.flatnonzero(A[i] == 1) for i in range(p)]
    q = deque([i for i in range(p) if indeg[i] == 0])
    seen = 0
    while q:
        i = q.popleft()
        seen += 1
        for j in children[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(int(j))
    return seen == p


def _parents_list(A: np.ndarray):
    return [np.flatnonzero(A[:, i] == 1) for i in range(A.shape[0])]


def _descendants(A: np.ndarray) -> np.ndarray:
    """desc[i, j] = 1 if a directed walk of length >= 1 goes from i to j."""
    p = A.shape[0]
    reach = A.astype(bool).copy()
    step = reach.copy()
    for _ in range(p - 1):
        step = step @ A.astype(bool)
        reach |= step
    np.fill_diagonal(reach, False)
    return reach


def d_separated(A: np.ndarray, x: int, y: int, Z) -> bool:
    """True if x and y are d-separated by Z in the DAG (or directed graph) A.

    Uses the ancestral-moralization criterion. Z must not contain x or y;
    if it does, the query is treated as separated.
    """
    Z = set(int(z) for z in Z)
    if x == y or x in Z or y in Z:
        return True
    p = A.shape[0]
    parents = _parents_list(A)
    ancestral = {x, y} | Z
    changed = True
    while changed:
        changed = False
        for node in list(ancestral):
            for par in parents[node]:
                par = int(par)
                if par not in ancestral:
                    ancestral.add(par)
                    changed = True
    nodes = sorted(ancestral)
    index = {n: i for i, n in enumerate(nodes)}
    m = len(nodes)
    und = np.zeros((m, m), dtype=bool)
    node_set = set(nodes)
    for i in nodes:
        pars = [int(j) for j in parents[i] if int(j) in node_set]
        ii = index[i]
        for j in pars:
            und[ii, index[j]] = True
            und[index[j], ii] = True
        for a in range(len(pars)):
            for b in range(a + 1, len(pars)):
                und[index[pars[a]], index[pars[b]]] = True
                und[index[pars[b]], index[pars[a]]] = True
    keep = [n for n in nodes if n not in Z]
    if x not in keep or y not in keep:
        return True
    seen = {x}
    q = deque([x])
    while q:
        u = q.popleft()
        ui = index[u]
        for v in keep:
            if und[ui, index[v]] and v not in seen:
                if v == y:
                    return False
                seen.add(v)
                q.append(v)
    return True


def sid(true_W: np.ndarray, est_W: np.ndarray, threshold: float = 0.0):
    """Structural intervention distance between two DAGs.

    Follows the parent-adjustment checks in Peters and Bühlmann: a pair
    (i, j) is incorrect when the estimated parents of i do not identify the
    effect of i on j in the true graph. A parent j of i is read as a null
    effect. Identical graphs score 0. Returns None if either graph is cyclic.
    """
    T = binarize(true_W, 0.0)
    E = binarize(est_W, threshold)
    if not is_acyclic(T) or not is_acyclic(E):
        return None
    p = T.shape[0]
    desc = _descendants(T)
    pa_true = [set(int(z) for z in row) for row in _parents_list(T)]
    pa_est = [set(int(z) for z in row) for row in _parents_list(E)]
    count = 0
    for i in range(p):
        pa_g = pa_true[i]
        pa_gp = pa_est[i]
        mutilated = T.copy()
        mutilated[i, :] = 0
        children = [int(c) for c in np.flatnonzero(T[i] == 1)]
        for j in range(p):
            if i == j:
                continue
            truth_null = not bool(desc[i, j])
            estimate_null = j in pa_gp
            if estimate_null and truth_null:
                continue
            if estimate_null and not truth_null:
                count += 1
                continue
            if pa_g == pa_gp:
                continue
            if not truth_null:
                causal_children = [c for c in children if c == j or bool(desc[c, j])]
                blocked = False
                for c in causal_children:
                    for z in pa_gp:
                        if z == c or bool(desc[c, z]):
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    count += 1
                    continue
            Z = [z for z in pa_gp if z != i and z != j]
            if not d_separated(mutilated, i, j, Z):
                count += 1
    return count


def varsortability(X: np.ndarray, W: np.ndarray, tol: float = 1e-9) -> float:
    """Fraction of directed paths along which marginal variance strictly increases.

    Reisach, Seiler, Weichwald (2021): a path i ~> j of any length contributes
    1 if Var(X_i) < Var(X_j), 1/2 if equal, and 0 otherwise. Paths are counted
    by the number of walks, matching the sum over E^k in their definition.
    Returns 1.0 on an empty graph.
    """
    p = W.shape[0]
    A = (np.abs(W) > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    var = np.var(X, axis=0, ddof=0)
    walks = A.copy()
    numerator = 0.0
    denominator = 0.0
    for _ in range(p - 1):
        ii, jj = np.nonzero(walks)
        if len(ii) == 0:
            break
        weight = walks[ii, jj]
        increase = np.where(
            var[ii] < var[jj] - tol,
            1.0,
            np.where(np.abs(var[ii] - var[jj]) <= tol, 0.5, 0.0),
        )
        numerator += float(np.sum(weight * increase))
        denominator += float(np.sum(weight))
        walks = walks @ A
    if denominator == 0.0:
        return 1.0
    return numerator / denominator


def self_check():
    """Small identities the metrics have to satisfy before any run is trusted."""
    # d-separation
    chain = np.zeros((3, 3), dtype=np.int8)
    chain[0, 1] = chain[1, 2] = 1
    assert d_separated(chain, 0, 2, []) is False
    assert d_separated(chain, 0, 2, [1]) is True
    collider = np.zeros((3, 3), dtype=np.int8)
    collider[0, 2] = collider[1, 2] = 1
    assert d_separated(collider, 0, 1, []) is True
    assert d_separated(collider, 0, 1, [2]) is False
    fork = np.zeros((3, 3), dtype=np.int8)
    fork[0, 1] = fork[0, 2] = 1
    assert d_separated(fork, 1, 2, []) is False
    assert d_separated(fork, 1, 2, [0]) is True

    # SHD
    true = chain.astype(float)
    assert shd(true, true) == 0
    rev = true.copy()
    rev[0, 1] = 0
    rev[1, 0] = 1
    assert shd(true, rev) == 1
    extra = true.copy()
    extra[0, 2] = 1
    assert shd(true, extra) == 1
    missing = true.copy()
    missing[1, 2] = 0
    assert shd(true, missing) == 1

    # SID(G, G) = 0, and a reversed edge on two nodes has SID 2.
    assert sid(true, true) == 0
    two = np.array([[0.0, 1.0], [0.0, 0.0]])
    two_rev = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert sid(two, two) == 0
    assert sid(two, two_rev) == 2

    # Varsortability is 1 when variance increases along the only edge.
    X = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 2.0]])
    W = np.array([[0.0, 1.0], [0.0, 0.0]])
    assert abs(varsortability(X, W) - 1.0) < 1e-9
    assert abs(varsortability(X[:, ::-1], W) - 0.0) < 1e-9
