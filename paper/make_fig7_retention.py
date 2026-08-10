r"""
make_fig7_retention.py -- gap retention versus disorder, both channels.

Answers the sharpest objection to the architecture: the clean-limit conversion
Delta_top/Delta = 0.98 sits against 0.12 measured in the only comparable
fabricated device (InAs-Pb tetrons, arXiv:2606.03884).

Two disorder channels are shown separately because they are not equivalent and
sweeping them together was actively misleading: bond disorder dominates. The
horizontal band marks the measured 12%, and where each curve crosses it is the
disorder level at which this model would reproduce that device's retention.

Points past the loss of the topological phase are dropped, not drawn. The
diagnostic is the Majorana splitting: exponentially small while the phase
exists, rising to the bulk scale once it is gone. Without that check the curve
turns non-monotonic at high disorder and the tail is meaningless.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tvqpu.lattice import REV21, MajoranaChannel

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})
C_MAIN, C_ALT, C_ACC, C_MUT = "#1f4e79", "#c1121f", "#2a9d8f", "#6c757d"

DELTA, VZ, N, SEEDS = 2.0, 3.95, 200, 24
P = replace(REV21, mu=0.0, v_z=VZ, delta=DELTA)
T = P.t


def realisation(sig, seed, mode):
    rng = np.random.default_rng(seed)
    sc = sig * T
    ch = MajoranaChannel(n_sites=N, params=P)
    dmu = rng.normal(0.0, sc, N)
    dt = rng.normal(0.0, sc, N - 1)
    if mode == "mu":
        ch = replace(ch, delta_mu=dmu)
    else:
        ch = replace(ch, delta_mu=dmu, delta_t=dt)
    ev = np.sort(np.abs(ch.spectrum()))
    return float(ev[0]), float(ev[2])


def curve(mode, sigmas):
    xs, ys, lo, hi = [], [], [], []
    for s in sigmas:
        out = [realisation(s, 1000 + k, mode) for k in range(SEEDS)]
        spl = np.median([o[0] for o in out])
        bulk = np.array([o[1] for o in out])
        med = float(np.median(bulk))
        if spl >= 0.1 * med:          # topological phase gone: not a gap
            break
        xs.append(s * T / DELTA)
        ys.append(med / DELTA)
        lo.append(np.percentile(bulk, 25) / DELTA)
        hi.append(np.percentile(bulk, 75) / DELTA)
    return map(np.array, (xs, ys, lo, hi))


def main():
    sig = np.concatenate([np.linspace(0.0, 0.15, 13), np.linspace(0.175, 0.45, 8)])
    fig, ax = plt.subplots(figsize=(3.35, 2.6))

    for mode, col, lab in (("mu", C_MAIN, r"electrostatic only ($\delta\mu$)"),
                           ("both", C_ALT, r"$\delta\mu$ + bond disorder $\delta t$")):
        x, y, l, h = curve(mode, sig)
        ax.fill_between(x, l, h, color=col, alpha=0.15, lw=0)
        ax.plot(x, y, color=col, lw=1.5, label=lab)
        print(f"  {mode:5} valid to dmu/Delta = {x[-1]:.2f}, "
              f"retention {y[-1]*100:.1f}%")

    # Both reference labels previously sat at x = 0.06, where the red curve is
    # still descending steeply and runs straight through them. Moved to the
    # right-hand half, which is empty below retention 0.35, and given white
    # backing so they stay legible where the blue curve passes.
    ax.axhline(0.12, color=C_MUT, ls="--", lw=1.0)
    ax.text(3.15, 0.135, r"InAs--Pb measured, 0.12", fontsize=6.4,
            color=C_MUT, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.85))
    ax.axhline(0.26, color=C_ACC, ls=":", lw=1.0)
    ax.text(3.15, 0.285, r"this architecture fails below", fontsize=6.4,
            color=C_ACC, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.85))

    ax.set_xlabel(r"disorder strength $\delta_{\rm rms}/\Delta$")
    ax.set_ylabel(r"retention $\Delta_{\rm top}/\Delta$")
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "fig7_retention.pdf")
    fig.savefig(OUT / "fig7_retention.png", dpi=200)
    plt.close(fig)
    print("  fig7_retention.pdf/.png")


if __name__ == "__main__":
    main()
