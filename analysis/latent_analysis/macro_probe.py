"""Quantitative probe: do A_psi macro-actions linearly encode net motion?

Reads ./figures/macro_action_tworoom.npz (macros: (N,8), disp: (N,2)) and fits a
linear regressor  macro -> (dx, dy)  with k-fold cross-validation. Reports R^2
(overall + per-axis) and a single-column figure with the Delta-x (top) and
Delta-y (bottom) predicted-vs-true scatters stacked vertically, each against the
ideal y=x. Turns the qualitative macro-action t-SNE into a quant+qual result: a
high CV R^2 means the 8-d macro-action *linearly* encodes the chunk's net motion;
CV vs train R^2 also reports overfitting.

Output is Overleaf-ready: a single-column-width vector PDF (embedded fonts,
rasterized points) plus a PNG preview. Pure sklearn/matplotlib on the saved
arrays — no torch, dataset, or GPU.

Usage
-----
python macro_probe.py                                   # ./figures/*.npz -> ./figures/*.pdf
python macro_probe.py --npz <path> --out <path.pdf> --cv 5
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

# Overleaf/CVPR-ready: embed TrueType fonts (avoid Type-3), print-sized fonts so a
# single-column \includegraphics[width=\linewidth] renders ~1:1 and stays readable.
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

_FIG = Path(__file__).resolve().parent / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(_FIG / "macro_action_tworoom.npz"))
    ap.add_argument("--out", default=str(_FIG / "macro_probe_tworoom.pdf"),
                    help="vector output (.pdf, import-ready); a .png preview is written alongside")
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.npz)
    X, Y = d["macros"], d["disp"]                # (N, 8) macros, (N, 2) (dx, dy)
    print(f"loaded {len(X)} (macro, displacement) pairs from {args.npz}")

    reg = LinearRegression()
    Y_cv = cross_val_predict(reg, X, Y, cv=args.cv)      # honest generalization estimate
    r2_overall = r2_score(Y, Y_cv)
    r2_axes = r2_score(Y, Y_cv, multioutput="raw_values")
    Y_tr = reg.fit(X, Y).predict(X)                      # train fit (overfit check)
    r2_train = r2_score(Y, Y_tr)

    print(f"{args.cv}-fold CV R^2 (macro -> net motion): "
          f"overall={r2_overall:.3f}  dx={r2_axes[0]:.3f}  dy={r2_axes[1]:.3f}")
    print(f"train R^2 = {r2_train:.3f}  (CV ~= train => negligible overfit)")

    # Single-column figure: dx (top) and dy (bottom) scatters stacked vertically.
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 5.8))
    for ax, j, name, r2 in [(axes[0], 0, r"$\Delta x$", r2_axes[0]),
                            (axes[1], 1, r"$\Delta y$", r2_axes[1])]:
        t, p = Y[:, j], Y_cv[:, j]
        lim = [float(min(t.min(), p.min())), float(max(t.max(), p.max()))]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.7, zorder=3)
        ax.scatter(t, p, s=4, alpha=0.35, edgecolors="none",
                   color="#1f77b4", rasterized=True)
        ax.set_title(f"{name}:  CV $R^2$ = {r2:.2f}")
        ax.set_xlabel(f"true {name} (env units)")
        ax.set_ylabel(f"predicted {name}")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout(pad=0.5)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")           # vector PDF (points rasterized)
    preview = out.with_suffix(".png")
    fig.savefig(preview, dpi=200, bbox_inches="tight")       # PNG preview
    print(f"saved {out}  (+ preview {preview})")


if __name__ == "__main__":
    main()
