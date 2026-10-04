"""A literal transformer forward pass with analytically fixed weights.

`experiments/realization.py` evaluates the *arithmetic* of one Poly-prox update
directly in numpy.  This module instead builds the residual-stream register
layout of the construction and instantiates every step as either a linear
attention head

    O(Z) = (Z Wq) (Z Wk)^T (Z Wv),

or a feedforward sublayer

    FF(Z) = relu(Z U1) U2 + [(Z G1) * (Z G2)] G3,

applied residually: Z <- Z + sum_h O_h(Z), then Z <- Z + FF(Z).  Only matrix
products, elementwise products and ReLU ever touch the data.  Every weight
matrix is a fixed pattern determined by p_max alone -- it does not depend on p,
on Sigma, on the mask, or on the controls.

Why the head form gives what the schedule needs
-----------------------------------------------
Writing P_A for the (width x p_max) matrix selecting register A's columns, a
head with Wq = P_A, Wk = P_B, Wv = P_C E_T adds to row i of register T

    sum_j <A_i, B_j> C_j  =  (A B^T C)_i .

Three cases carry the construction:

  * B = id (the one-hot identity register) gives B^T C = C, so the head is the
    matrix product A C.  This yields S^2, Q S and Sigma W.
  * A = id and C = id gives row i equal to sum_j B_ji e_j, the TRANSPOSE of B.
    This is how Q^T is formed.
  * A, B the two token-type flags reduce the head to a cross-token sum into the
    control token, or a broadcast of a control scalar back to every row.

The third case is the one a feedforward sublayer cannot do, because it moves
information between tokens.

Masking
-------
No bounded-input soft-threshold is needed.  X = M * (Sigma W - Sigma) is masked
by construction, HS = W * Q^T is masked because W is, and W itself is masked by
hypothesis, so V = W - gamma X - 2 sigma HS lies in W_M; soft-thresholding maps
0 to 0, hence W^+ = soft(V) is already masked.  The construction therefore
places no bound on any input.
"""

from __future__ import annotations

import numpy as np

MAT_REGS = ["id", "Sigma", "M", "W", "B", "Q", "S", "SW", "ST", "HS", "X", "V", "Wn"]
SCALAR_REGS = [
    "fctl", "fvar",                        # token-type flags
    "lam", "alpha", "rho", "gamma", "bfl",  # the five controls
    "gl", "ga", "gr", "brho",              # gamma*lam, gamma*alpha, gamma*rho, b*rho
    "h", "hplus", "vsig", "d",             # constraint value, post-update value, multiplier, diagonal
    "gl_b", "gam_b", "vsig_b",             # broadcast copies of gamma*lam, gamma, vsig
]


class Layout:
    def __init__(self, p_max: int):
        self.p_max = p_max
        self.mat_off = {n: i * p_max for i, n in enumerate(MAT_REGS)}
        base = len(MAT_REGS) * p_max
        self.sca_off = {n: base + i for i, n in enumerate(SCALAR_REGS)}
        self.width = base + len(SCALAR_REGS)

    def P(self, name):                      # width x p_max : select matrix register
        M = np.zeros((self.width, self.p_max))
        o = self.mat_off[name]
        M[o:o + self.p_max, :] = np.eye(self.p_max)
        return M

    def E(self, name):                      # p_max x width : write matrix register
        return self.P(name).T

    def p(self, name):                      # width x 1 : select scalar register
        v = np.zeros((self.width, 1))
        v[self.sca_off[name], 0] = 1.0
        return v

    def e(self, name):                      # 1 x width : write scalar register
        return self.p(name).T

    def bcast(self, name):                  # width x p_max : scalar replicated across a row
        return self.p(name) @ np.ones((1, self.p_max))


class Head:
    def __init__(self, Wq, Wk, Wv, tag=""):
        self.Wq, self.Wk, self.Wv, self.tag = Wq, Wk, Wv, tag
        self.key_dim = Wq.shape[1]

    def __call__(self, Z):
        return (Z @ self.Wq) @ (Z @ self.Wk).T @ (Z @ self.Wv)


class FF:
    def __init__(self, U1=None, U2=None, G1=None, G2=None, G3=None, tag=""):
        self.U1, self.U2, self.G1, self.G2, self.G3, self.tag = U1, U2, G1, G2, G3, tag
        self.n_relu = 0 if U1 is None else U1.shape[1]
        self.n_bilin = 0 if G1 is None else G1.shape[1]

    def __call__(self, Z):
        out = np.zeros_like(Z)
        if self.U1 is not None:
            out = out + np.maximum(Z @ self.U1, 0.0) @ self.U2
        if self.G1 is not None:
            out = out + ((Z @ self.G1) * (Z @ self.G2)) @ self.G3
        return out


class Layer:
    def __init__(self, heads=(), ff=None, tag=""):
        self.heads, self.ff, self.tag = list(heads), ff, tag

    def __call__(self, Z):
        if self.heads:
            Z = Z + sum(h(Z) for h in self.heads)
        if self.ff is not None:
            Z = Z + self.ff(Z)
        return Z


def _relu_linear(L):
    """Realize the exact linear update Z += Z L as relu(Z[L,-L])[I;-I]."""
    k = L.shape[1]
    U1 = np.concatenate([L, -L], axis=1)
    U2 = np.concatenate([np.eye(k), -np.eye(k)], axis=0)
    return U1, U2


def build(p_max: int):
    """Fixed weights for the block at this p_max. Returns (layers, layout, N, s)."""
    lay = Layout(p_max)
    N = 1 << int(np.ceil(np.log2(max(p_max, 2))))
    s = int(np.log2(N))
    ones = np.ones((1, p_max))
    layers = []

    def mm(A, B, C, T, coef=1.0):
        """T += coef * A B^T C."""
        return Head(lay.P(A), lay.P(B), coef * (lay.P(C) @ lay.E(T)), tag=f"{T}+={A}{B}^T{C}")

    def gather(src, dst):
        """control token += sum over variable tokens of scalar src."""
        return Head(lay.p("fctl"), lay.p("fvar"), lay.p(src) @ lay.e(dst), tag=f"{dst}+=sum {src}")

    def spread(src, dst):
        """every variable token += the control token's scalar src."""
        return Head(lay.p("fvar"), lay.p("fctl"), lay.p(src) @ lay.e(dst), tag=f"{dst}<-{src}")

    def lin(*terms):
        L = np.zeros((lay.width, lay.width))
        for c, A, B in terms:
            L += c * (A @ B)
        return L

    # 1. B = W*W, S = B/N, and the four control products on the control token.
    G1 = np.concatenate([lay.P("W"), lay.p("gamma"), lay.p("gamma"), lay.p("gamma"), lay.p("bfl")], axis=1)
    G2 = np.concatenate([lay.P("W"), lay.p("lam"), lay.p("alpha"), lay.p("rho"), lay.p("rho")], axis=1)
    G3 = np.concatenate([lay.E("B") + lay.E("S") / N, lay.e("gl"), lay.e("ga"), lay.e("gr"), lay.e("brho")], axis=0)
    layers.append(Layer(ff=FF(G1=G1, G2=G2, G3=G3, tag="B=W*W, S=B/N, control products"), tag="init"))

    # 2. s doubling layers: S <- 2S + S^2, Q <- Q + S + Q S.
    def doubling(tag):
        heads = [mm("S", "id", "S", "V"), mm("Q", "id", "S", "X")]
        L = lin((1.0, lay.P("S"), lay.E("Q")), (1.0, lay.P("X"), lay.E("Q")),
                (1.0, lay.P("S"), lay.E("S")), (1.0, lay.P("V"), lay.E("S")),
                (-1.0, lay.P("X"), lay.E("X")), (-1.0, lay.P("V"), lay.E("V")))
        U1, U2 = _relu_linear(L)
        return Layer(heads=heads, ff=FF(U1=U1, U2=U2, tag="Q+=S+QS; S+=S+S^2; clear scratch"), tag=tag)

    for j in range(s):
        layers.append(doubling(f"squaring {j + 1}"))

    # 3. d_i = <e_i, S_i> = S_ii.
    layers.append(Layer(ff=FF(G1=lay.P("id"), G2=lay.P("S"), G3=np.ones((p_max, 1)) @ lay.e("d"),
                              tag="d=diag(S)"), tag="diagonal"))

    # 4. h = sum_i d_i ; ST = Q^T ; SW = Sigma W ; broadcast gamma.
    #    then X = M*(SW - Sigma), HS = W*ST, vsig = ga + gr*h.
    heads = [gather("d", "h"),
             Head(lay.P("id"), lay.P("Q"), lay.P("id") @ lay.E("ST"), tag="ST=Q^T"),
             mm("Sigma", "id", "W", "SW"),
             spread("gamma", "gam_b")]
    G1 = np.concatenate([lay.P("M"), lay.P("W"), lay.p("gr")], axis=1)
    G2 = np.concatenate([lay.P("SW") - lay.P("Sigma"), lay.P("ST"), lay.p("h")], axis=1)
    G3 = np.concatenate([lay.E("X"), lay.E("HS"), lay.e("vsig")], axis=0)
    U1, U2 = _relu_linear(lin((1.0, lay.p("ga"), lay.e("vsig"))))
    layers.append(Layer(heads=heads, ff=FF(U1=U1, U2=U2, G1=G1, G2=G2, G3=G3,
                                           tag="X, HS, vsig"), tag="assemble gradient"))

    # 5. broadcast vsig and gamma*lam, then V = W - gamma X - 2 vsig HS.
    heads = [spread("vsig", "vsig_b"), spread("gl", "gl_b")]
    G1 = np.concatenate([lay.P("X"), lay.P("HS")], axis=1)
    G2 = np.concatenate([lay.bcast("gam_b"), lay.bcast("vsig_b")], axis=1)
    G3 = np.concatenate([-lay.E("V"), -2.0 * lay.E("V")], axis=0)
    U1, U2 = _relu_linear(lin((1.0, lay.P("W"), lay.E("V"))))
    layers.append(Layer(heads=heads, ff=FF(U1=U1, U2=U2, G1=G1, G2=G2, G3=G3,
                                           tag="V=W-gamma X-2 vsig HS"), tag="proximal input"))

    # 6. Wn = soft_{gamma lambda}(V) = relu(V - gl) - relu(-V - gl).
    A1 = lay.P("V") - lay.bcast("gl_b")
    A2 = -lay.P("V") - lay.bcast("gl_b")
    U1 = np.concatenate([A1, A2], axis=1)
    U2 = np.concatenate([lay.E("Wn"), -lay.E("Wn")], axis=0)
    layers.append(Layer(ff=FF(U1=U1, U2=U2, tag="soft threshold"), tag="prox"))

    # 7. clear scratch; B = Wn*Wn, S = B/N for the second squaring pass.
    L = lin(*[(-1.0, lay.P(r), lay.E(r)) for r in ("Q", "S", "B", "V", "X", "ST", "HS", "SW")],
            (-1.0, lay.p("d"), lay.e("d")))
    U1, U2 = _relu_linear(L)
    layers.append(Layer(ff=FF(U1=U1, U2=U2, G1=lay.P("Wn"), G2=lay.P("Wn"),
                              G3=lay.E("B") + lay.E("S") / N, tag="reset; B,S from Wn"),
                        tag="restart"))

    # 8. second squaring pass.
    for j in range(s):
        layers.append(doubling(f"squaring' {j + 1}"))

    # 9. d_i = S_ii again.
    layers.append(Layer(ff=FF(G1=lay.P("id"), G2=lay.P("S"), G3=np.ones((p_max, 1)) @ lay.e("d"),
                              tag="d=diag(S)"), tag="diagonal'"))

    # 10. h+ = sum_i d_i ; alpha += b rho h+ ; W <- Wn ; clear the rest.
    heads = [gather("d", "hplus")]
    L = lin((-1.0, lay.P("W"), lay.E("W")), (1.0, lay.P("Wn"), lay.E("W")),
            *[(-1.0, lay.P(r), lay.E(r)) for r in ("Q", "S", "B", "Wn")],
            *[(-1.0, lay.p(r), lay.e(r)) for r in ("d", "gl", "ga", "gr", "gl_b", "gam_b", "vsig", "vsig_b")])
    U1, U2 = _relu_linear(L)
    layers.append(Layer(heads=heads, ff=FF(U1=U1, U2=U2, G1=lay.p("brho"), G2=lay.p("hplus"),
                                           G3=lay.e("alpha"), tag="alpha update; W<-Wn"),
                        tag="finish"))
    return layers, lay, N, s


def encode(Sigma, M, W, alpha, lam, rho, gamma, b, lay):
    """Build Z_0 from the problem data: p variable tokens plus one control token."""
    p = W.shape[0]
    pm = lay.p_max
    Z = np.zeros((p + 1, lay.width))
    def put(name, A):
        o = lay.mat_off[name]
        Z[:p, o:o + pm] = A
    put("id", np.eye(p, pm))
    put("Sigma", np.pad(Sigma, ((0, 0), (0, pm - p))))
    put("M", np.pad(M, ((0, 0), (0, pm - p))))
    put("W", np.pad(W, ((0, 0), (0, pm - p))))
    Z[:p, lay.sca_off["fvar"]] = 1.0
    Z[p, lay.sca_off["fctl"]] = 1.0
    for name, val in (("lam", lam), ("alpha", alpha), ("rho", rho), ("gamma", gamma), ("bfl", b)):
        Z[p, lay.sca_off[name]] = float(val)
    return Z


def decode(Z, p, lay):
    o = lay.mat_off["W"]
    W = Z[:p, o:o + lay.p_max][:, :p]
    return W, float(Z[p, lay.sca_off["alpha"]])


def run(Sigma, M, W, alpha, lam, rho, gamma, b, p_max, layers=None, lay=None):
    """One update, computed by the actual attention/feedforward forward pass."""
    if layers is None:
        layers, lay, _, _ = build(p_max)
    Z = encode(Sigma, M, W, alpha, lam, rho, gamma, b, lay)
    for L in layers:
        Z = L(Z)
    return decode(Z, W.shape[0], lay)


def cost(p_max):
    """Depth, per-layer head count, key dimension, width, and unit counts."""
    layers, lay, N, s = build(p_max)
    heads = [len(L.heads) for L in layers]
    keys = [h.key_dim for L in layers for h in L.heads] or [0]
    relu = [L.ff.n_relu for L in layers if L.ff is not None]
    bilin = [L.ff.n_bilin for L in layers if L.ff is not None]
    mags = set()
    for L in layers:
        for h in L.heads:
            for Mx in (h.Wq, h.Wk, h.Wv):
                mags.update(np.unique(np.abs(Mx[Mx != 0])).tolist())
        if L.ff is not None:
            for Mx in (L.ff.U1, L.ff.U2, L.ff.G1, L.ff.G2, L.ff.G3):
                if Mx is not None:
                    mags.update(np.unique(np.abs(Mx[Mx != 0])).tolist())
    return {
        "p_max": p_max, "N": N, "s": s,
        "depth": len(layers), "depth_formula_2s+8": 2 * s + 8,
        "max_heads_per_layer": max(heads), "max_key_dim": max(keys),
        "width": lay.width, "width_formula_13pmax+18": 13 * p_max + 18,
        "max_relu_units": max(relu), "max_bilinear_units": max(bilin),
        "nonzero_magnitudes": sorted(mags),
    }
