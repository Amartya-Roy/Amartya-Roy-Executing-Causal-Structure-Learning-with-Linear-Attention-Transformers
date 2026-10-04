"""Baselines named in the planned evaluation.

PC and GES come from causal-learn and return CPDAGs. Sort-regress is the
variance-sorting least-squares procedure of Reisach, Seiler, and Weichwald.
NOTEARS is the exponential-constraint augmented Lagrangian solved by L-BFGS,
matching Zheng et al. The polynomial proximal solver lives in solver.py.
"""

from __future__ import annotations

import numpy as np

from experiments.solver import lbfgs_al, score


def sortnregress(X: np.ndarray) -> np.ndarray:
    """Regress each variable on the nodes with smaller marginal variance."""
    m, p = X.shape
    order = np.argsort(np.var(X, axis=0, ddof=0))
    W = np.zeros((p, p))
    for pos, j in enumerate(order):
        parents = order[:pos]
        if len(parents) == 0:
            continue
        design = X[:, parents]
        coef, *_ = np.linalg.lstsq(design, X[:, j], rcond=None)
        W[parents, j] = coef
    return W


def _cpdag_to_adj(graph: np.ndarray) -> np.ndarray:
    """causal-learn encoding to a 0/1 matrix.

    graph[i, j] == -1 and graph[j, i] == 1 means i -> j.
    graph[i, j] == graph[j, i] == -1 means an undirected edge, stored in both
    directions so the SHD routine can see it.
    """
    p = graph.shape[0]
    A = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if graph[i, j] == -1 and graph[j, i] == 1:
                A[i, j] = 1.0
            elif graph[i, j] == -1 and graph[j, i] == -1 and i < j:
                A[i, j] = 1.0
                A[j, i] = 1.0
    return A


def pc_estimate(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    from causallearn.search.ConstraintBased.PC import pc

    cg = pc(X, alpha=alpha, indep_test="fisherz", stable=True, show_progress=False, verbose=False)
    return _cpdag_to_adj(np.asarray(cg.G.graph))


def ges_estimate(X: np.ndarray) -> np.ndarray:
    from causallearn.search.ScoreBased.GES import ges

    record = ges(X, score_func="local_score_BIC")
    return _cpdag_to_adj(np.asarray(record["G"].graph))


def notears_expm(Sigma: np.ndarray, lam: float = 0.1):
    return lbfgs_al(Sigma, lam=lam, kind="expm")


def notears_poly_lbfgs(Sigma: np.ndarray, lam: float = 0.1):
    return lbfgs_al(Sigma, lam=lam, kind="poly")


def baseline_score(Sigma, W, lam):
    return score(Sigma, W, lam)


def dagma_estimate(X: np.ndarray, lambda1: float = 0.03, threshold: float = 0.3) -> np.ndarray:
    """Linear DAGMA with the package defaults. W[i, j] is the weight of i -> j."""
    from dagma.linear import DagmaLinear

    model = DagmaLinear(loss_type="l2")
    W = model.fit(np.asarray(X, dtype=np.float64), lambda1=lambda1, w_threshold=threshold)
    W = np.asarray(W, dtype=np.float64)
    np.fill_diagonal(W, 0.0)
    return W


def golem_ev(
    X: np.ndarray,
    lambda_1: float = 2e-2,
    lambda_2: float = 5.0,
    num_iter: int = 20000,
    lr: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """GOLEM-EV in PyTorch, matching Ng, Ghassami, and Zhang (2020).

    The loss is the equal-variance Gaussian likelihood, an L1 term, and
    tr(exp(B odot B)) - p. The SEM is X = X B + noise, so B[i, j] is i -> j.
    The published training length is 10^5 Adam steps; `num_iter` is the
    budget used here.
    """
    import torch

    torch.manual_seed(seed)
    data = torch.tensor(np.asarray(X, dtype=np.float64))
    _n, d = data.shape
    raw = torch.zeros(d, d, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)
    eye = torch.eye(d, dtype=torch.float64)
    for _ in range(int(num_iter)):
        opt.zero_grad()
        B = raw - torch.diag(torch.diag(raw))
        resid = data - data @ B
        likelihood = 0.5 * d * torch.log(torch.sum(resid * resid) + 1e-12)
        likelihood = likelihood - torch.slogdet(eye - B)[1]
        h = torch.trace(torch.matrix_exp(B * B)) - d
        loss = likelihood + lambda_1 * torch.sum(torch.abs(B)) + lambda_2 * h
        loss.backward()
        opt.step()
    B = raw.detach() - torch.diag(torch.diag(raw.detach()))
    return B.cpu().numpy()
