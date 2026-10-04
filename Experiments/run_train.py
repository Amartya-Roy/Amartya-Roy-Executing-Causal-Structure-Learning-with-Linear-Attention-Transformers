"""Experiment 3: train a weight-tied block to execute one proximal step.

The supervision is the polynomial update itself, on stage checkpoints and on
random states. One weight set is trained at p=5 with rows padded to p_max=10
and evaluated at p=10 with no retraining. The collapsed prompt drops the
W and alpha registers. An unrolled proximal map with a learned threshold is
the LISTA-style baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from experiments.metrics import shd
from experiments.realization import one_update
from experiments.sem import covariance, sample_dag, simulate_sem
from experiments.solver import proximal_al

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

P_MAX = 10
N = 16  # 2^ceil(log2(10)), one N for every p <= p_max
LAM = 0.1


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad_rows(A, p_max):
    """Pad the last axis (a row) up to p_max. A is (..., p)."""
    pad = p_max - A.shape[-1]
    if pad == 0:
        return A
    return np.pad(A, [(0, 0)] * (A.ndim - 1) + [(0, pad)])


def random_transition(p, rng):
    graph = "er" if rng.random() < 0.5 else "sf"
    true_W = sample_dag(p, float(rng.choice([1, 2, 4])), graph, rng)
    m = int(rng.choice([200, 500, 1000]))
    noise = "gaussian" if rng.random() < 0.5 else "laplace"
    X = simulate_sem(true_W, m, noise, rng, standardize=bool(rng.random() < 0.5))
    Sigma = covariance(X)
    W = rng.normal(scale=0.2, size=(p, p))
    np.fill_diagonal(W, 0.0)
    mask = np.ones((p, p))
    np.fill_diagonal(mask, 0.0)
    alpha = float(rng.uniform(0.0, 2.0))
    rho = float(rng.choice([0.1, 1.0, 10.0]))
    gamma = float(rng.uniform(0.02, 0.2) / max(np.linalg.norm(Sigma, 2), 1e-6))
    W_next, alpha_next, _, _ = one_update(Sigma, W, mask, alpha, rho, gamma, LAM, 1.0, N)
    return pack(Sigma, mask, W, alpha, rho, gamma, W_next, alpha_next, true_W)


def solver_transitions(p, rng, n_keep=12):
    """States visited by the proximal solver, labeled by one exact step."""
    true_W = sample_dag(p, 2.0, "er" if rng.random() < 0.5 else "sf", rng)
    X = simulate_sem(true_W, 500, "gaussian", rng, standardize=False)
    Sigma = covariance(X)
    mask = np.ones((p, p))
    np.fill_diagonal(mask, 0.0)
    # Replay a short proximal loop and keep a spread of states.
    from experiments.solver import _smooth, prox_map

    W = np.zeros((p, p))
    alpha = 0.0
    rho = 1.0
    gamma = 1.0 / max(float(np.linalg.norm(Sigma, 2)), 1e-6)
    h_prev = np.inf
    kept = []
    for _outer in range(6):
        for _inner in range(25):
            val, grad, h, _gh = _smooth(Sigma, W, alpha, rho, N, "poly")
            if not np.isfinite(val):
                break
            gtry = gamma
            W_new = W
            accepted = False
            for _bt in range(20):
                W_new = prox_map(W, grad, gtry, LAM, mask)
                dW = W_new - W
                val_new, _, _, _ = _smooth(Sigma, W_new, alpha, rho, N, "poly")
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
            W_next, alpha_next, _, _ = one_update(Sigma, W, mask, alpha, rho, gtry, LAM, 0.0, N)
            kept.append(pack(Sigma, mask, W, alpha, rho, gtry, W_next, alpha_next, true_W))
            gap = float(np.linalg.norm(W_new - W)) / max(gtry, 1e-16)
            W = W_new
            gamma = min(gtry * 1.2, 10.0 / max(float(np.linalg.norm(Sigma, 2)), 1e-6))
            if gap < 1e-6:
                break
        _val, grad, h, _gh = _smooth(Sigma, W, alpha, rho, N, "poly")
        if h <= 1e-8 or rho >= 1e12:
            break
        if h > 0.25 * h_prev:
            rho = min(rho * 10.0, 1e12)
        alpha = alpha + rho * h
        h_prev = h
    if len(kept) > n_keep:
        idx = np.linspace(0, len(kept) - 1, n_keep).astype(int)
        kept = [kept[i] for i in idx]
    return kept, Sigma, mask, true_W


def pack(Sigma, mask, W, alpha, rho, gamma, W_next, alpha_next, true_W):
    from experiments.hgrad import h_and_grad

    h_val, gh = h_and_grad(W, N)
    return {
        "h": np.float32(h_val),
        "grad_h": pad_rows(gh, P_MAX),
        "Sigma": pad_rows(Sigma, P_MAX),
        "mask": pad_rows(mask, P_MAX),
        "W": pad_rows(W, P_MAX),
        "alpha": np.float32(alpha),
        "rho": np.float32(rho),
        "gamma": np.float32(gamma),
        "lam": np.float32(LAM),
        "W_next": pad_rows(W_next, P_MAX),
        "alpha_next": np.float32(alpha_next),
        "true_W": true_W,
        "p": Sigma.shape[0],
    }


class TiedAttention(nn.Module):
    """Weight-tied attention block.

    use_memory  -- W and alpha are given as inputs (dedicated input coordinates).
    persistent  -- the output *writes back into* those registers (W_hat = W + delta,
                   alpha_hat = alpha + head). With persistent=False the same inputs are
                   available but the output is produced from scratch, so the model has to
                   re-create W and alpha instead of carrying them; parameter count,
                   inputs and target are identical.
    aux         -- extra heads for intermediate supervision (grad h(W) rows and h(W)).
    """

    def __init__(self, p_max, d=64, repeats=4, kind="linear", use_memory=True,
                 persistent=True, aux=False):
        super().__init__()
        self.p_max = p_max
        self.repeats = repeats
        self.use_memory = use_memory
        self.persistent = persistent
        self.aux = aux
        n_rows = 3 if use_memory else 2
        n_scalars = 4 if use_memory else 3  # alpha is a memory register
        self.in_proj = nn.Linear(n_rows * p_max + n_scalars, d)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.kind = kind
        self.ff1 = nn.Linear(d, 2 * d)
        self.ff2 = nn.Linear(2 * d, d)
        self.out = nn.Linear(d, p_max)
        self.alpha_head = nn.Linear(d, 1)
        # Zero the residual writes so repeated application starts at the identity.
        nn.init.zeros_(self.wv.weight)
        nn.init.zeros_(self.ff2.weight)
        nn.init.zeros_(self.ff2.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        if aux:
            self.gradh_head = nn.Linear(d, p_max)
            self.h_head = nn.Linear(d, 1)

    def forward(self, Sigma, mask, W, alpha, rho, gamma, lam, return_aux=False):
        # Sigma, mask, W: B, p, p_max (rows already padded on the last axis)
        parts = [Sigma, mask]
        scalars = [rho, gamma, lam]
        if self.use_memory:
            parts.append(W)
            scalars = [alpha, rho, gamma, lam]
        feats = torch.cat(parts, dim=-1)
        s = torch.stack(scalars, dim=-1)[:, None, :].expand(-1, Sigma.shape[1], -1)
        z = self.in_proj(torch.cat([feats, s], dim=-1))
        for _ in range(self.repeats):
            q, k, v = self.wq(z), self.wk(z), self.wv(z)
            logits = q @ k.transpose(-1, -2) / (z.shape[-1] ** 0.5)
            if self.kind == "linear":
                att = logits / z.shape[1]
            else:
                att = torch.softmax(logits, dim=-1)
            z = z + att @ v
            z = z + self.ff2(torch.relu(self.ff1(z)))
        delta = self.out(z)
        if self.use_memory and self.persistent:
            W_hat = W + delta
            alpha_hat = alpha + self.alpha_head(z.mean(dim=1)).squeeze(-1)
        else:
            W_hat = delta
            alpha_hat = self.alpha_head(z.mean(dim=1)).squeeze(-1)
        W_hat = W_hat * mask
        if return_aux:
            gh = self.gradh_head(z) if self.aux else None
            hh = self.h_head(z.mean(dim=1)).squeeze(-1) if self.aux else None
            return W_hat, alpha_hat, {"grad_h": gh, "h": hh}
        return W_hat, alpha_hat


class LearnedThreshold(nn.Module):
    """Tied proximal map: a learned linear step on the least-squares gradient,
    then a learned soft threshold. No acyclicity gradient and no attention."""

    def __init__(self, p_max, hidden=160, repeats=4):
        super().__init__()
        self.p_max = p_max
        self.repeats = repeats
        self.net = nn.Sequential(
            nn.Linear(3 * p_max, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, p_max),
        )
        self.log_tau = nn.Parameter(torch.tensor(-2.0))

    def forward(self, Sigma, mask, W, alpha, rho, gamma, lam):
        eye = torch.eye(self.p_max, device=Sigma.device, dtype=Sigma.dtype)
        # Sigma is B, p, p_max with only the first p columns nonzero. The
        # least-squares gradient needs the p x p block.
        p = Sigma.shape[1]
        Sig = Sigma[:, :, :p]
        Wc = W[:, :, :p]
        grad = torch.matmul(Sig, Wc - eye[:p, :p])
        grad_pad = Sigma.new_zeros(Sigma.shape)
        grad_pad[:, :, :p] = grad
        inp = torch.cat([W, Sigma, grad_pad], dim=-1)
        raw = W + self.net(inp)
        tau = torch.nn.functional.softplus(self.log_tau)
        W_hat = torch.sign(raw) * torch.clamp(raw.abs() - tau, min=0) * mask
        return W_hat, alpha


def batch_of(records, idx, dev):
    take = [records[i] for i in idx]
    def stack(key):
        return torch.tensor(np.stack([r[key] for r in take]), dtype=torch.float32, device=dev)
    return {
        "Sigma": stack("Sigma"),
        "mask": stack("mask"),
        "W": stack("W"),
        "alpha": stack("alpha"),
        "rho": stack("rho"),
        "gamma": stack("gamma"),
        "lam": stack("lam"),
        "W_next": stack("W_next"),
        "alpha_next": stack("alpha_next"),
    }


def train_model(model, records, dev, epochs=12, batch=64, lr=3e-4):
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = len(records)
    history = []
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total = 0.0
        count = 0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            if len(idx) < 8:
                continue
            b = batch_of(records, idx, dev)
            W_hat, a_hat = model(b["Sigma"], b["mask"], b["W"], b["alpha"], b["rho"], b["gamma"], b["lam"])
            # Supervise the active columns. Padded columns are zero in the target.
            w_loss = ((W_hat - b["W_next"]) ** 2).mean()
            a_loss = ((a_hat - b["alpha_next"]) ** 2).mean()
            loss = w_loss + 0.05 * a_loss
            if not torch.isfinite(loss):
                opt.zero_grad()
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
            count += len(idx)
        history.append(total / max(count, 1))
    return history


@torch.no_grad()
def one_step_error(model, records, dev):
    model.eval()
    errs = []
    for start in range(0, len(records), 64):
        b = batch_of(records, range(start, min(start + 64, len(records))), dev)
        W_hat, _a = model(b["Sigma"], b["mask"], b["W"], b["alpha"], b["rho"], b["gamma"], b["lam"])
        diff = (W_hat - b["W_next"]).cpu().numpy()
        target = b["W_next"].cpu().numpy()
        source = b["W"].cpu().numpy()
        for i in range(diff.shape[0]):
            p = records[start + i]["p"]
            num = np.linalg.norm(diff[i, :p, :p])
            step = np.linalg.norm(target[i, :p, :p] - source[i, :p, :p]) + 1e-8
            errs.append(num / step)
    return float(np.median(errs)), float(np.mean(errs))


def build_dataset(n_random, n_solver, p, seed):
    rng = np.random.default_rng(seed)
    records = [random_transition(p, rng) for _ in range(n_random)]
    for _ in range(n_solver):
        kept, *_rest = solver_transitions(p, rng)
        records.extend(kept)
    return records


def unroll_eval(model, dev, p, n_problems, seed):
    """Replay solver controls. The model carries its own W and alpha."""
    rng = np.random.default_rng(seed)
    model.eval()
    curves = []
    for _ in range(n_problems):
        true_W = sample_dag(p, 2.0, "er", rng)
        X = simulate_sem(true_W, 800, "gaussian", rng, False)
        Sigma = covariance(X)
        solved = proximal_al(Sigma, lam=LAM, kind="poly")
        # A short explicit trajectory with the same controls the solver uses
        # roughly: fixed gamma, rho increasing when h stalls. We compare the
        # model's iterate to the exact update's iterate under shared controls.
        mask = np.ones((p, p))
        np.fill_diagonal(mask, 0.0)
        gamma = 0.2 / max(float(np.linalg.norm(Sigma, 2)), 1e-6)
        rho = 1.0
        W_exact = np.zeros((p, p))
        a_exact = 0.0
        W_model = np.zeros((p, p))
        a_model = 0.0
        series = []
        for ell in range(12):
            W_exact, a_exact, h, _ = one_update(
                Sigma, W_exact, mask, a_exact, rho, gamma, LAM, 1.0 if ell % 4 == 3 else 0.0, N
            )
            b = 1.0 if ell % 4 == 3 else 0.0
            batch = {
                "Sigma": pad_rows(Sigma, P_MAX)[None],
                "mask": pad_rows(mask, P_MAX)[None],
                "W": pad_rows(W_model, P_MAX)[None],
                "alpha": np.array([a_model], dtype=np.float32),
                "rho": np.array([rho], dtype=np.float32),
                "gamma": np.array([gamma], dtype=np.float32),
                "lam": np.array([LAM], dtype=np.float32),
            }
            tensors = {k: torch.tensor(v, dtype=torch.float32, device=dev) for k, v in batch.items()}
            with torch.no_grad():
                W_hat, a_hat = model(
                    tensors["Sigma"], tensors["mask"], tensors["W"],
                    tensors["alpha"], tensors["rho"], tensors["gamma"], tensors["lam"],
                )
            W_model = W_hat[0, :p, :p].cpu().numpy()
            a_model = float(a_hat[0].cpu())
            series.append({
                "ell": ell,
                "dist_exact": float(np.linalg.norm(W_model - W_exact)),
                "dist_solver": float(np.linalg.norm(W_model - solved["W"])),
                "shd_true": int(shd(true_W, W_model, 0.3)),
                "shd_exact": int(shd(true_W, W_exact, 0.3)),
                "exact_h": float(h),
            })
            if b == 1.0 and h > 1e-8:
                rho = min(rho * 10.0, 1e6)
        curves.append(series)
    # Average across problems.
    L = len(curves[0])
    averaged = []
    for ell in range(L):
        averaged.append({
            "ell": ell,
            "dist_exact": float(np.mean([c[ell]["dist_exact"] for c in curves])),
            "dist_solver": float(np.mean([c[ell]["dist_solver"] for c in curves])),
            "shd_true": float(np.mean([c[ell]["shd_true"] for c in curves])),
            "shd_exact": float(np.mean([c[ell]["shd_exact"] for c in curves])),
        })
    return averaged


def main():
    dev = device()
    print("device", dev, flush=True)
    print("building data", flush=True)
    train = build_dataset(1500, 80, p=5, seed=0)
    test5 = build_dataset(200, 0, p=5, seed=1)
    test10 = build_dataset(120, 0, p=10, seed=2)
    print(f"train {len(train)}  test5 {len(test5)}  test10 {len(test10)}", flush=True)

    specs = {
        "linear_memory": TiedAttention(P_MAX, kind="linear", use_memory=True),
        "softmax_memory": TiedAttention(P_MAX, kind="softmax", use_memory=True),
        "linear_collapsed": TiedAttention(P_MAX, kind="linear", use_memory=False),
        "learned_threshold": LearnedThreshold(P_MAX),
    }
    report = {}
    for name, model in specs.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"training {name} ({n_params} params)", flush=True)
        history = train_model(model, train, dev)
        med5, mean5 = one_step_error(model, test5, dev)
        med10, mean10 = one_step_error(model, test10, dev)
        print(f"  loss {history[0]:.4f} -> {history[-1]:.4f}  "
              f"rel p5 {med5:.3f}  p10 {med10:.3f}", flush=True)
        curve = unroll_eval(model, dev, p=5, n_problems=12, seed=3)
        report[name] = {
            "params": n_params,
            "train_loss": history,
            "one_step_median_p5": med5,
            "one_step_mean_p5": mean5,
            "one_step_median_p10": med10,
            "one_step_mean_p10": mean10,
            "unroll_p5": curve,
        }
        torch.save(model.state_dict(), OUT / f"{name}.pt")
    path = OUT / "trained_execution.json"
    path.write_text(json.dumps(report, indent=2))
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
