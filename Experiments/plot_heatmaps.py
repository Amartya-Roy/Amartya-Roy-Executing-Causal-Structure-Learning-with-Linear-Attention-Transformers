"""Heatmaps of decoded edge matrices, in the layout of the planned figure.

Each panel is |W|. Red circles mark edges of the true graph. The right-hand
column is the solver's own fixed point, kept separate from the true edges.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.metrics import shd
from experiments.realization import one_update
from experiments.run_train import (
    LAM,
    N,
    P_MAX,
    TiedAttention,
    pad_rows,
)
from experiments.sem import covariance, sample_dag, simulate_sem
from experiments.solver import _smooth, prox_map, proximal_al

ROOT = Path(__file__).resolve().parents[1]
RES = Path(__file__).resolve().parent / "results"
FIG = ROOT / "figures"


def load(name, kind, memory):
    model = TiedAttention(P_MAX, kind=kind, use_memory=memory)
    state = torch.load(RES / f"{name}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def instance(p, seed):
    rng = np.random.default_rng(seed)
    true_W = sample_dag(p, 2.0, "er", rng)
    X = simulate_sem(true_W, 800, "gaussian", rng, standardize=False)
    Sigma = covariance(X)
    solved = proximal_al(Sigma, lam=LAM, kind="poly")
    return true_W, Sigma, solved["W"]


def pick_instance(p, seed0, trials=8):
    best = None
    for seed in range(seed0, seed0 + trials):
        true_W, Sigma, W_ref = instance(p, seed)
        n_edges = int(np.sum(np.abs(true_W) > 0))
        if n_edges < 3:
            continue
        err = shd(true_W, W_ref, 0.3)
        score = (err, -n_edges)
        if best is None or score < best[0]:
            best = (score, seed, true_W, Sigma, W_ref)
        if err == 0 and n_edges >= 4:
            break
    _score, seed, true_W, Sigma, W_ref = best
    return seed, true_W, Sigma, W_ref


@torch.no_grad()
def step_model(model, Sigma, W, alpha, rho, gamma):
    p = Sigma.shape[0]
    mask = np.ones((p, p))
    np.fill_diagonal(mask, 0.0)
    batch = {
        "Sigma": pad_rows(Sigma, P_MAX)[None],
        "mask": pad_rows(mask, P_MAX)[None],
        "W": pad_rows(W, P_MAX)[None],
        "alpha": np.array([alpha], np.float32),
        "rho": np.array([rho], np.float32),
        "gamma": np.array([gamma], np.float32),
        "lam": np.array([LAM], np.float32),
    }
    tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in batch.items()}
    W_hat, a_hat = model(
        tensors["Sigma"], tensors["mask"], tensors["W"],
        tensors["alpha"], tensors["rho"], tensors["gamma"], tensors["lam"],
    )
    return W_hat[0, :p, :p].numpy(), float(a_hat[0])


def unroll(model, Sigma, n_steps=12):
    """Shared controls with the exact block, as in unroll_eval."""
    p = Sigma.shape[0]
    mask = np.ones((p, p))
    np.fill_diagonal(mask, 0.0)
    gamma = 0.2 / max(float(np.linalg.norm(Sigma, 2)), 1e-6)
    rho = 1.0
    W = np.zeros((p, p))
    alpha = 0.0
    W_exact = np.zeros((p, p))
    a_exact = 0.0
    frames = []
    exact = []
    for ell in range(n_steps):
        b = 1.0 if ell % 4 == 3 else 0.0
        W_exact, a_exact, h, _hp = one_update(
            Sigma, W_exact, mask, a_exact, rho, gamma, LAM, b, N
        )
        W, alpha = step_model(model, Sigma, W, alpha, rho, gamma)
        frames.append(W.copy())
        exact.append(W_exact.copy())
        if b == 1.0 and h > 1e-8:
            rho = min(rho * 10.0, 1e6)
    return frames, exact


def solver_snapshots(Sigma, steps):
    """Accepted proximal states at the requested step counts, plus the final W."""
    p = Sigma.shape[0]
    degree = 1 if p <= 1 else 1 << int(np.ceil(np.log2(p)))
    mask = np.ones((p, p))
    np.fill_diagonal(mask, 0.0)
    W = np.zeros((p, p))
    alpha = 0.0
    rho = 1.0
    h_prev = np.inf
    eye_norm = float(np.linalg.norm(Sigma, 2))
    gamma = 1.0 / max(eye_norm, 1e-6)
    wanted = set(steps)
    found = {}
    step = 0
    for _outer in range(12):
        stalled = False
        for _inner in range(400):
            val, grad, h, _gh = _smooth(Sigma, W, alpha, rho, degree, "poly")
            if not np.isfinite(val):
                stalled = True
                break
            accepted = False
            gtry = gamma
            W_new = W
            for _bt in range(25):
                W_new = prox_map(W, grad, gtry, LAM, mask)
                dW = W_new - W
                val_new, _, _, _ = _smooth(Sigma, W_new, alpha, rho, degree, "poly")
                if not np.isfinite(val_new):
                    gtry *= 0.5
                    continue
                rhs = val + float(np.sum(grad * dW)) + (0.5 / gtry) * float(np.sum(dW * dW))
                if val_new <= rhs + 1e-8 * (1.0 + abs(val)):
                    accepted = True
                    break
                gtry *= 0.5
            if not accepted:
                stalled = True
                break
            gap = float(np.linalg.norm(W_new - W)) / max(gtry, 1e-16)
            W = W_new
            gamma = min(gtry * 1.2, 10.0 / max(eye_norm, 1e-6))
            step += 1
            if step in wanted and step not in found:
                found[step] = W.copy()
            if gap < 1e-6 or step >= max(wanted):
                stalled = True
                break
        _val, grad, h, _gh = _smooth(Sigma, W, alpha, rho, degree, "poly")
        if stalled or h <= 1e-8 or rho >= 1e16:
            break
        if h > 0.25 * h_prev:
            rho = min(rho * 10.0, 1e16)
        alpha = alpha + rho * h
        h_prev = h
    return found, W


def draw(ax, M, true_W):
    show = np.abs(np.asarray(M, dtype=float))
    vmax = max(float(np.max(show)), 1e-8)
    ax.imshow(show, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
    ii, jj = np.nonzero(np.abs(true_W) > 0)
    size = 36 if true_W.shape[0] <= 6 else 14
    ax.scatter(
        jj, ii, s=size, c="#e23b3b", edgecolors="none",
        linewidths=0, zorder=3,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.4)
    return vmax


def main():
    softmax = load("softmax_memory", "softmax", True)
    collapsed = load("linear_collapsed", "linear", False)

    print("picking p=5", flush=True)
    seed5, true5, Sigma5, ref5 = pick_instance(5, seed0=22, trials=1)
    print("picking p=10", flush=True)
    seed10, true10, Sigma10, ref10 = pick_instance(10, seed0=83, trials=1)
    frames, _exact5 = unroll(softmax, Sigma5, n_steps=12)
    without_frames, _ = unroll(collapsed, Sigma5, n_steps=12)
    gen_frames, _exact10 = unroll(softmax, Sigma10, n_steps=12)
    without = without_frames[-1]
    with_mem = frames[-1]
    gen = gen_frames[-1]
    snaps, _final = solver_snapshots(Sigma5, steps=(1, 5, 40))

    print(
        "p5 seed", seed5, "edges", int(np.sum(np.abs(true5) > 0)),
        "shd", shd(true5, ref5, 0.3),
        "p10 seed", seed10, "edges", int(np.sum(np.abs(true10) > 0)),
        "shd", shd(true10, ref10, 0.3),
    )
    for name, M in (
        ("gen", gen), ("ref10", ref10),
        ("without", without), ("with", with_mem), ("ref5", ref5),
        ("s1", snaps[1]), ("s5", snaps[5]), ("s40", snaps[40]),
    ):
        print(name, "max", round(float(np.max(np.abs(M))), 4))

    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.05, 4.35))
    gs = fig.add_gridspec(
        3, 5,
        width_ratios=[1, 1, 1, 0.10, 1.05],
        wspace=0.28,
        hspace=0.48,
        left=0.14, right=0.86, top=0.90, bottom=0.08,
    )

    def panel(r, c, M, true_W, title):
        ax = fig.add_subplot(gs[r, c])
        draw(ax, M, true_W)
        ax.set_title(title, pad=3)
        return ax

    panel(0, 0, gen, true10, r"trained, test $p{=}10$")
    ax_ref_top = panel(0, 4, ref10, true10, "estimated graph")
    panel(1, 0, without, true5, "without registers")
    panel(1, 1, with_mem, true5, "with registers")
    panel(1, 4, ref5, true5, "estimated graph")
    panel(2, 0, snaps[1], true5, "update 1")
    panel(2, 1, snaps[5], true5, "update 5")
    panel(2, 2, snaps[40], true5, "update 40")
    ax_ref_bot = panel(2, 4, ref5, true5, "estimated graph")

    # Vertical rule between the readouts and W_ref, as in the OT figures.
    x_line = ax_ref_top.get_position().x0 - 0.02
    fig.add_artist(plt.Line2D(
        [x_line, x_line],
        [ax_ref_bot.get_position().y0, ax_ref_top.get_position().y1],
        transform=fig.transFigure,
        color="black",
        linewidth=0.8,
        clip_on=False,
    ))

    fig.text(0.015, 0.76, "(1) generalization", rotation=90, va="center", ha="left", fontsize=8)
    fig.text(0.015, 0.48, "(2) prompt", rotation=90, va="center", ha="left", fontsize=8)
    fig.text(0.015, 0.20, "(3) proximal updates", rotation=90, va="center", ha="left", fontsize=8)

    cax = fig.add_axes([0.885, 0.08, 0.015, 0.78])
    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 1.0), cmap="viridis"),
        cax=cax,
    )
    cax.tick_params(labelsize=6)
    cax.set_ylabel("edge strength", fontsize=7)

    path = FIG / "heatmaps.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig("/tmp/heatmaps.png", dpi=140, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
