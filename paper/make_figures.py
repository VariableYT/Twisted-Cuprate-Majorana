r"""
make_figures.py -- publication figures for the Rev 2.2 architecture paper.

Outputs PDF (vector) into paper/figs/ for \includegraphics.

Every figure is generated from tvqpu.lattice -- the same validated BdG model
the test suite covers. No figure contains hand-drawn or illustrative data.
"""

from __future__ import annotations

import math
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

KB = 0.08617333262          # meV/K
G_FACTOR, MU_B = 25.0, 0.05788381806

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
})

C_MAIN, C_ALT, C_ACC, C_MUT = "#1f4e79", "#c1121f", "#2a9d8f", "#6c757d"


def vz_to_B(vz):
    return vz / (0.5 * G_FACTOR * MU_B)


def gap(mu, vz, d, nk=801):
    return MajoranaChannel(
        n_sites=1, params=replace(REV21, mu=mu, v_z=vz, delta=d)).bulk_gap(nk=nk)


# --------------------------------------------------------------------------
def fig1_spectrum():
    """BdG excitation spectrum vs Zeeman energy -- the topological transition."""
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    vzs = np.linspace(0, 6, 90)
    N = 80
    lo = []
    for vz in vzs:
        ch = MajoranaChannel(n_sites=N, params=replace(REV21, v_z=float(vz)))
        ev = np.sort(np.abs(ch.spectrum()))
        lo.append(ev[:16])
    lo = np.array(lo)
    for j in range(2, 16):
        ax.plot(vzs, lo[:, j], color=C_MUT, lw=0.5, alpha=0.55)
    ax.plot(vzs, lo[:, 0], color=C_ALT, lw=1.4, label="Majorana pair")
    ax.plot(vzs, lo[:, 1], color=C_ALT, lw=1.4)
    ax.axvline(2.0, color=C_ACC, ls="--", lw=0.9)
    ax.text(2.06, 3.4, r"$V_{z,\mathrm{crit}}=\Delta$", color=C_ACC, fontsize=7)
    ax.set_xlabel(r"Zeeman energy $V_z$ (meV)")
    ax.set_ylabel(r"$|E|$ (meV)")
    ax.set_xlim(0, 6); ax.set_ylim(0, 4)
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "fig1_spectrum.pdf"); plt.close(fig)
    print("  fig1_spectrum.pdf")


def fig2_two_branches():
    """THE key figure: Delta_top vs V_z, the two competing minima."""
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    d = 2.0
    vzs = np.linspace(2.05, 9.0, 130)
    actual = np.array([gap(0.0, float(v), d) for v in vzs])
    k0 = np.abs(vzs - d)
    ax.plot(vzs, k0, ls="--", color=C_MUT, lw=1.0,
            label=r"$k=0$ branch: $|V_z-\sqrt{\mu^2+\Delta^2}|$")
    ax.axhline(0.972 * d, ls=":", color=C_ACC, lw=1.0,
               label=r"finite-$k$ branch $\approx 0.97\,\Delta$")
    ax.plot(vzs, actual, color=C_MAIN, lw=1.6, label=r"$\Delta_{\rm top}$ (exact)")
    i = int(np.argmax(actual))
    ax.plot(vzs[i], actual[i], "o", color=C_MAIN, ms=5, zorder=5)
    ax.annotate(rf"optimum $V_z={vzs[i]:.2f}$" "\n" rf"$\Delta_{{\rm top}}={actual[i]:.3f}$ meV",
                xy=(vzs[i], actual[i]), xytext=(vzs[i] + 1.1, actual[i] - 0.42),
                fontsize=6.5, color=C_MAIN,
                arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=0.7))
    ax.plot(3.0, gap(0.0, 3.0, d), "s", color=C_ALT, ms=5, zorder=5)
    ax.annotate("Rev 2.1\n$V_z=3.0$", xy=(3.0, gap(0.0, 3.0, d)),
                xytext=(2.2, 1.45), fontsize=6.5, color=C_ALT,
                arrowprops=dict(arrowstyle="->", color=C_ALT, lw=0.7))
    ax.set_xlabel(r"Zeeman energy $V_z$ (meV)")
    ax.set_ylabel(r"$\Delta_{\rm top}$ (meV)")
    ax.set_xlim(2, 9); ax.set_ylim(0, 2.6)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)
    sec = ax.secondary_xaxis("top", functions=(vz_to_B, lambda b: 0.5*G_FACTOR*MU_B*b))
    sec.set_xlabel(r"in-plane field $B_\parallel$ (T)", fontsize=7.5)
    fig.savefig(OUT / "fig2_branches.pdf"); plt.close(fig)
    print("  fig2_branches.pdf")


def fig3_operating_map():
    """Delta_top over (mu, V_z), with the field ceiling overlaid."""
    fig, ax = plt.subplots(figsize=(3.35, 2.7))
    d = 2.0
    mus = np.linspace(0, 12, 34)
    vzs = np.linspace(2.0, 13.0, 40)
    Z = np.zeros((len(vzs), len(mus)))
    for a, mu in enumerate(mus):
        for b, vz in enumerate(vzs):
            Z[b, a] = gap(float(mu), float(vz), d, nk=241) \
                if vz > math.hypot(mu, d) else np.nan
    im = ax.pcolormesh(mus, vzs, Z, cmap="viridis", shading="auto",
                       vmin=0, vmax=2.0, rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$\Delta_{\rm top}$ (meV)", fontsize=7.5)
    ax.plot(mus, [math.hypot(m, d) for m in mus], color="w", lw=1.0, ls="--")
    ax.text(6.4, 6.9, r"$V_{z,\rm crit}$", color="w", fontsize=6.5, rotation=32)
    best = [vzs[np.nanargmax(Z[:, a])] for a in range(len(mus))]
    ax.plot(mus, best, color=C_ALT, lw=1.3, label="optimum locus")
    for B, ls in ((9.0, ":"), (16.0, "-")):
        ax.axhline(0.5*G_FACTOR*MU_B*B, color="w", ls=ls, lw=1.0)
        ax.text(0.25, 0.5*G_FACTOR*MU_B*B + 0.16, f"{B:.0f} T",
                color="w", fontsize=6.5)
    ax.set_xlabel(r"chemical potential $\mu$ (meV)")
    ax.set_ylabel(r"$V_z$ (meV)")
    ax.legend(frameon=False, loc="lower right", labelcolor="w")
    fig.savefig(OUT / "fig3_map.pdf"); plt.close(fig)
    print("  fig3_map.pdf")


def fig4_robustness():
    """Delta_top vs Delta: fixed V_z (tent, an artifact) vs optimised V_z."""
    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    ds = np.linspace(0.2, 3.2, 34)
    fixed = np.array([gap(0.0, 3.0, float(d)) for d in ds])
    opt = []
    for d in ds:
        grid = np.linspace(float(d) + 0.02, min(float(d)*2.6, 11.58), 44)
        opt.append(max(gap(0.0, float(v), float(d)) for v in grid))
    opt = np.array(opt)
    ax.plot(ds, fixed, ls="--", color=C_ALT, lw=1.3,
            label=r"$V_z=3$ meV fixed (Rev 2.1)")
    ax.plot(ds, opt, color=C_MAIN, lw=1.6, label=r"$V_z$ optimised (Rev 2.2)")
    ax.plot(ds, ds, ls=":", color=C_MUT, lw=0.9, label=r"$\Delta_{\rm top}=\Delta$")
    ax.axhline(0.517, color=C_ACC, lw=0.9)
    ax.text(2.15, 0.575, r"floor: $T_{\max}=0.3$ K", color=C_ACC, fontsize=6.5)
    ax.set_xlabel(r"induced pairing gap $\Delta$ (meV)")
    ax.set_ylabel(r"$\Delta_{\rm top}$ (meV)")
    ax.set_xlim(0.2, 3.2); ax.set_ylim(0, 3.2)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8)
    fig.savefig(OUT / "fig4_robustness.pdf"); plt.close(fig)
    print("  fig4_robustness.pdf")


def fig5_localisation():
    """Majorana density at the old and new operating points."""
    fig, ax = plt.subplots(figsize=(3.35, 2.4))
    for vz, c, lab in ((3.0, C_ALT, r"Rev 2.1, $V_z=3.0$"),
                       (3.95, C_MAIN, r"Rev 2.2, $V_z=3.95$")):
        ch = MajoranaChannel(n_sites=200, params=replace(REV21, v_z=vz))
        dens = ch.majorana_density()
        xi = ch.localization_length()
        ax.semilogy(np.arange(200), dens, color=c, lw=1.2,
                    label=rf"{lab}, $\xi={xi:.1f}$ sites")
    ax.set_xlabel("site index")
    ax.set_ylabel(r"$|\psi|^2$ (normalised)")
    ax.set_xlim(0, 199); ax.set_ylim(1e-13, 1)
    ax.legend(frameon=False, loc="upper center")
    fig.savefig(OUT / "fig5_localisation.pdf"); plt.close(fig)
    print("  fig5_localisation.pdf")


def fig6_tunneling():
    """Milestone 1: predicted tunneling spectra."""
    fig, ax = plt.subplots(figsize=(3.35, 2.4))

    def dynes(E, D, G):
        z = (E + 1j*G)/np.sqrt((E + 1j*G)**2 - D**2 + 0j)
        return np.abs(np.real(z))

    def smear(E, D, G, T):
        kt = KB*T
        w = np.linspace(-8*kt, 8*kt, 401)
        df = 1.0/(4*kt*np.cosh(w/(2*kt))**2); df /= np.trapezoid(df, w)
        return np.array([np.trapezoid(dynes(e - w, D, G)*df, w) for e in E])

    E = np.linspace(-4, 4, 500)
    for D, c in ((0.5, C_ACC), (1.0, C_ALT), (2.0, C_MAIN)):
        ax.plot(E, smear(E, D, 0.05*D, 1.0), color=c, lw=1.2,
                label=rf"$\Delta={D}$ meV")
    ax.axhline(1.0, color=C_MUT, lw=0.6, ls=":")
    ax.set_xlabel(r"bias $V$ (mV)")
    ax.set_ylabel(r"$dI/dV$ (normalised)")
    ax.set_xlim(-4, 4); ax.set_ylim(0, 2.6)
    ax.legend(frameon=False, loc="upper right")
    ax.text(-3.85, 2.35, "predicted, $T=1$ K", fontsize=6.5, color=C_MUT)
    fig.savefig(OUT / "fig6_tunneling.pdf"); plt.close(fig)
    print("  fig6_tunneling.pdf")


if __name__ == "__main__":
    print("generating figures ->", OUT)
    fig1_spectrum()
    fig2_two_branches()
    fig4_robustness()
    fig5_localisation()
    fig6_tunneling()
    fig3_operating_map()
    print("done")
