"""Paper schematic of the exact Poly-prox / transformer correspondence.

The terrain and path are illustrative, not measured optimization losses.
Both panels use exactly the same geometry and checkpoint coordinates.
Run from the project root: .venv/bin/python experiments/plot_dynamics_correspondence.py
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import PolyCollection
import numpy as np

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                     "mathtext.fontset": "cm", "pdf.fonttype": 42,
                     "ps.fonttype": 42, "svg.fonttype": "none"})


def terrain(x, y):
    return (0.95 * np.exp(-((x + .85)**2 / .48 + (y - .48)**2 / .55))
            + .64 * np.exp(-((x - .8)**2 / .6 + (y - .75)**2 / .48))
            - .7 * np.exp(-((x - .62)**2 / .5 + (y + .62)**2 / .48))
            - .37 * np.exp(-((x + .78)**2 / .6 + (y + .45)**2 / .5)))


def project(x, y, z):
    """Shared orthographic view; a vector mesh avoids 3D axes clipping."""
    return .94*x + .34*y, -.08*x + .33*y + .68*z


def make_figure():
    # Native single-column dimensions for the two-column AISTATS layout.
    fig = plt.figure(figsize=(3.3, 2.8), facecolor="white")
    xs, ys = np.meshgrid(np.linspace(-2, 2, 53), np.linspace(-1.55, 1.55, 41))
    zs = terrain(xs, ys)
    t = np.linspace(0, 1, 160)
    # One shared illustrative path: its copy below encodes the same states.
    px = -1.35 + 2.0*t + .14*np.sin(2*np.pi*t)
    py = .35 - .96*t + .24*np.sin(np.pi*t)
    pz = terrain(px, py) + .025
    checkpoints = [0, 159]
    warm = LinearSegmentedColormap.from_list("warm", ["#b42b82", "#e86fa0", "#f5a877", "#ffd477", "#fff1a2"])
    cool = LinearSegmentedColormap.from_list("cool", ["#2447a6", "#367bd0", "#35add0", "#50c9ae", "#a6e2ad"])
    ux,uy = project(xs,ys,zs)
    cells = []
    for i in range(xs.shape[0]-1):
        for j in range(xs.shape[1]-1):
            k = [(i,j),(i,j+1),(i+1,j+1),(i+1,j)]
            cells.append((np.mean([.34*xs[a,b]-.94*ys[a,b] for a,b in k]),
                          [(ux[a,b],uy[a,b]) for a,b in k],
                          np.mean([zs[a,b] for a,b in k])))
    cells.sort(key=lambda c:c[0])
    axes = []
    for bottom, cmap in [(.535, warm), (.035, cool)]:
        ax = fig.add_axes([.035, bottom, .94, .43])
        mesh = PolyCollection([c[1] for c in cells],
                              facecolors=cmap(Normalize(-.72,1.0)([c[2] for c in cells])),
                              edgecolors=(.12,.18,.25,.50), linewidths=.19,
                              antialiaseds=True)
        ax.add_collection(mesh)
        ax.set(xlim=(-2.54,2.54), ylim=(-.91,1.08))
        ax.set_axis_off()
        axes.append(ax)

    # Only panel, point, and mathematical labels belong inside the image.
    fig.text(.02, .94, "(a)", fontsize=9, color="#282536")
    fig.text(.02, .44, "(b)", fontsize=9, color="#17334c")

    # Project and overlay the path so it is never hidden by the surface mesh.
    fig.canvas.draw()
    for panel, ax in enumerate(axes):
        qx,qy = project(px, py, pz)
        xy = fig.transFigure.inverted().transform(ax.transData.transform(np.c_[qx,qy]))
        line = plt.Line2D(xy[:,0], xy[:,1], transform=fig.transFigure,
                          color="#152f46", lw=1.55, solid_capstyle="round", zorder=20)
        line.set_path_effects([pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])
        fig.add_artist(line)
        for a,b in [(25,34),(72,81),(122,131)]:
            fig.add_artist(FancyArrowPatch(xy[a], xy[b], transform=fig.transFigure,
                           arrowstyle="-|>", mutation_scale=7, lw=.85,
                           color="#152f46", zorder=23))
        labels = ([r"$(W_\ell,\alpha_\ell)$",r"$(W_{\ell+1},\alpha_{\ell+1})$"] if panel==0 else
                  [r"$Z_\ell$",r"$Z_{\ell+1}$"])
        for idx,label in zip(checkpoints,labels):
            xx,yy = xy[idx]
            fig.add_artist(plt.Line2D([xx],[yy],transform=fig.transFigure, marker="o",
                           markersize=4, markerfacecolor="#152f46", markeredgecolor="white",
                           markeredgewidth=.8, zorder=24))
            dx,dy = (-.025,.067) if idx==0 else (.15,.037)
            fig.text(xx+dx,yy+dy,label,ha="center",va="center",fontsize=9,color="#142c43",
                     zorder=25,bbox=dict(boxstyle="round,pad=.10",facecolor="white",edgecolor="none",alpha=.94))
        mid = xy[74]
        op = r"$\mathcal{U}_{c_\ell}$" if panel==0 else r"$\mathcal{T}_{\theta,c_\ell}$"
        fig.text(mid[0]+.16,mid[1]+.055,op,fontsize=10,ha="center",color="#142c43",
                 bbox=dict(boxstyle="round,pad=.10",facecolor="white",edgecolor="none",alpha=.94),zorder=25)
    for ext in ("pdf", "svg", "png"):
        path = FIG / f"dynamics_correspondence.{ext}"
        fig.savefig(path, dpi=450, facecolor="white", bbox_inches=None)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
