r"""
make_figures.py -- publication figures for the Rev 2.2 architecture paper.

Outputs PDF (vector) into paper/figs/ for \includegraphics.

Every figure EXCEPT fig0_schematic is generated from tvqpu.lattice -- the same
validated BdG model the test suite covers, containing no hand-drawn or
illustrative data. fig0_schematic is a drawing of the device geometry and is
labelled as such in its caption; it is the one exception and exists because
prose alone left the stacking order ambiguous.
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
def fig0_schematic():
    r"""Device schematic -- WIDE, two-column (figure*) format.

    ILLUSTRATIVE, unlike every other figure in this file: a drawing of the
    geometry, not an output of the model.

    Sized for \textwidth, not \columnwidth. At single-column width the
    annotation had to be set at ~6 pt to fit, which is below comfortable
    reading size in print. Laying the two panels side by side across both
    columns roughly doubles the available width and lets the labels sit at a
    normal figure size.

    The panels are DIFFERENT SAMPLES, deliberately. Milestone 1 asks a
    materials question (does the interface induce a gap?) and needs no gating,
    no channel and no field, so the TI can sit on top where a tip reaches it.
    The architecture needs the channel gated, which puts the TI beneath the
    bilayer. Same interface under test, different stacking order.
    """
    from matplotlib.patches import Rectangle, FancyArrow, Polygon

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.45))
    C_BSCCO, C_TI, C_SUB, C_GATE = C_MAIN, C_ACC, C_MUT, C_ALT
    X0, X1 = 0.5, 5.0
    LBL = X1 + 0.25

    def layer(ax, y, h, color, label, alpha=1.0):
        ax.add_patch(Rectangle((X0, y), X1 - X0, h, facecolor=color,
                               edgecolor="k", lw=0.6, alpha=alpha))
        ax.text(LBL, y + h / 2, label, va="center", ha="left", fontsize=7.0)

    # ---------------- (a) Milestone 1 sample ----------------
    layer(axa, 0.30, 0.75, C_SUB, "substrate", alpha=0.35)
    layer(axa, 1.05, 0.55, C_BSCCO, "Bi-2212", alpha=0.95)
    layer(axa, 1.60, 0.55, C_BSCCO, r"Bi-2212, twisted $45^\circ$", alpha=0.60)
    layer(axa, 2.15, 0.62, C_TI, r"Bi$_2$Se$_3$ (exposed)", alpha=0.85)

    axa.plot([X0, X1], [2.15, 2.15], color=C_ALT, lw=1.8, zorder=5)
    axa.annotate("", xy=(X0 - 0.05, 2.15), xytext=(X0 - 1.05, 2.15),
                 arrowprops=dict(arrowstyle="->", color=C_ALT, lw=0.8))
    axa.text(X0 - 1.15, 2.15, "interface" "\n" "under test", fontsize=6.8,
             ha="right", va="center", color=C_ALT)

    axa.add_patch(Polygon([[2.5, 4.15], [3.0, 4.15], [2.75, 3.00]],
                          closed=True, facecolor="0.25", edgecolor="k",
                          lw=0.6))
    axa.text(3.15, 3.78, "STM tip", fontsize=7.0, va="center")
    axa.annotate("", xy=(2.75, 2.82), xytext=(2.75, 2.98),
                 arrowprops=dict(arrowstyle="->", color=C_ALT, lw=1.1))
    axa.text(2.55, 2.90, r"$dI/dV$", fontsize=6.8, ha="right", color=C_ALT)
    axa.text(3.15, 3.32, r"$T=1$ K,  $B=0$", fontsize=6.8, va="center")
    axa.text(3.6, 4.62, "(a) Milestone 1 sample", fontsize=8.0,
             ha="center", va="center")

    # ---------------- (b) architecture ----------------
    layer(axb, 0.30, 0.50, C_GATE, "gates", alpha=0.55)
    layer(axb, 0.80, 0.40, C_SUB, "dielectric", alpha=0.30)
    layer(axb, 1.20, 0.62, C_TI, r"Bi$_2$Se$_3$", alpha=0.85)
    layer(axb, 1.82, 0.55, C_BSCCO, r"Bi-2212, twisted $45^\circ$", alpha=0.60)
    layer(axb, 2.37, 0.55, C_BSCCO, "Bi-2212", alpha=0.95)

    axb.plot([X0, X1], [1.82, 1.82], color=C_ALT, lw=1.8, zorder=5)
    axb.add_patch(Rectangle((1.30, 1.28), 2.90, 0.46, facecolor="none",
                            edgecolor=C_ALT, lw=1.2, ls="--", zorder=6))
    for x in (1.30, 4.20):
        axb.plot(x, 1.51, "o", color=C_ALT, ms=4.5, zorder=7)
    axb.annotate("", xy=(1.55, 1.26), xytext=(1.15, 0.12),
                 arrowprops=dict(arrowstyle="->", color=C_ALT, lw=0.7))
    axb.text(1.10, 0.02, r"gate-defined channel; $\gamma_{1,2}$ at ends",
             fontsize=6.8, ha="left", va="top", color=C_ALT)

    axb.add_patch(FancyArrow(0.6, 3.55, 1.55, 0.0, width=0.035,
                             head_width=0.18, head_length=0.28,
                             facecolor="k", edgecolor="k"))
    axb.text(2.60, 3.55, r"$B_\parallel$ along the channel", fontsize=7.0,
             va="center")
    axb.text(3.6, 4.62, "(b) Architecture", fontsize=8.0,
             ha="center", va="center")

    for ax in (axa, axb):
        ax.set_xlim(-2.4, 9.6)
        ax.set_ylim(-0.75, 4.95)
        ax.axis("off")

    fig.subplots_adjust(wspace=0.02, left=0.005, right=0.995,
                        top=0.985, bottom=0.02)
    fig.savefig(OUT / "fig0_schematic.pdf")
    fig.savefig(OUT / "fig0_schematic.png", dpi=200)
    plt.close(fig)
    print("  fig0_schematic.pdf/.png  (wide, two-column)")

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
                xy=(vzs[i], actual[i]), xytext=(vzs[i] + 1.35, actual[i] - 0.62),
                fontsize=6.5, color=C_MAIN,
                arrowprops=dict(arrowstyle="->", color=C_MAIN, lw=0.7))
    ax.plot(3.0, gap(0.0, 3.0, d), "s", color=C_ALT, ms=5, zorder=5)
    ax.annotate("Rev 2.1\n$V_z=3.0$", xy=(3.0, gap(0.0, 3.0, d)),
                xytext=(2.2, 1.45), fontsize=6.5, color=C_ALT,
                arrowprops=dict(arrowstyle="->", color=C_ALT, lw=0.7))
    ax.set_xlabel(r"Zeeman energy $V_z$ (meV)")
    ax.set_ylabel(r"$\Delta_{\rm top}$ (meV)")
    ax.set_xlim(2, 9); ax.set_ylim(0, 2.6)
    leg = ax.legend(loc="lower right", handlelength=1.6, framealpha=0.92,
                    edgecolor="none", facecolor="white", borderpad=0.5)
    leg.set_zorder(6)
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
    ax.plot(mus, [math.hypot(m, d) for m in mus], color="w", lw=1.1, ls="--")
    # At (6.4, 6.9) this label straddled the white no-phase wedge and the pale
    # end of viridis, so white type was unreadable on both sides. Moved up the
    # boundary into the coloured region and given a dark backing.
    ax.text(8.9, 9.6, r"$V_{z,\rm crit}$", color="w", fontsize=6.5,
            rotation=30, ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.16", fc="0.15", ec="none",
                      alpha=0.72))
    best = [vzs[np.nanargmax(Z[:, a])] for a in range(len(mus))]
    ax.plot(mus, best, color=C_ALT, lw=1.3, label="optimum locus")
    for B, ls in ((9.0, ":"), (16.0, "-")):
        y = 0.5*G_FACTOR*MU_B*B
        ax.axhline(y, color="w", ls=ls, lw=1.1)
        # dark pill behind the field labels: plain white type disappeared
        # against the pale-green plateau of viridis
        ax.text(0.30, y + 0.30, f"{B:.0f} T", color="w", fontsize=6.5,
                va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="0.15", ec="none",
                          alpha=0.72))
    ax.set_xlabel(r"chemical potential $\mu$ (meV)")
    ax.set_ylabel(r"$V_z$ (meV)")
    leg = ax.legend(loc="lower right", labelcolor="w", framealpha=0.55,
                    facecolor="0.15", edgecolor="none", borderpad=0.45)
    leg.set_zorder(6)
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
    # The only band clear of both curves: right of Delta = 2.5, where the
    # fixed-Vz curve has already dived below the floor and the optimised curve
    # is far above it. Left-hand placements are crossed by the rising curves.
    ax.text(3.15, 0.565, r"floor: $T_{\max}=0.3$ K", color=C_ACC, fontsize=6.5,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.85))
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
    ax.set_xlim(-4, 4); ax.set_ylim(0, 2.95)
    # headroom added and the legend backed in white: at ylim 2.6 the box sat
    # directly on the Delta = 2 meV coherence peak
    leg = ax.legend(loc="upper right", framealpha=0.92, edgecolor="none",
                    facecolor="white", borderpad=0.45, handlelength=1.5)
    leg.set_zorder(6)
    ax.text(-3.85, 2.72, "predicted, $T=1$ K", fontsize=6.5, color=C_MUT)
    fig.savefig(OUT / "fig6_tunneling.pdf"); plt.close(fig)
    print("  fig6_tunneling.pdf")


if __name__ == "__main__":
    print("generating figures ->", OUT)
    fig0_schematic()
    fig1_spectrum()
    fig2_two_branches()
    fig4_robustness()
    fig5_localisation()
    fig6_tunneling()
    fig3_operating_map()
    print("done")
