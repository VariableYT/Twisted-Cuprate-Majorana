r"""
disorder_retention.py -- how much of the induced gap survives as topological
gap, as a function of electrostatic disorder.

WHY THIS EXISTS
---------------
Microsoft's InAs-Pb tetron (arXiv:2606.03884) measures an induced gap
Delta_ind = 570 ueV degrading to a topological gap Delta_T ~ 70 ueV: a
retention of about 12%. Our clean-limit design point retains 97%
(Delta = 2.000 -> Delta_top = 1.944 meV). That gap between 97% and 12% is the
single sharpest objection to this architecture, and this script measures where
our model sits as disorder is turned up.

WHAT IT DOES NOT DO
-------------------
It does NOT reverse-engineer Microsoft's loss coefficients. Their paper reports
one ratio; the loss has at least three mechanisms (electrostatic disorder,
sub-band occupation, orbital depairing) and one number cannot determine three
parameters. Any such split would be assumed, and the assumption would set the
answer.

This model is a single-band 1D chain. Two independent disorder channels are
available and they are NOT equivalent:

    mu-only   chemical-potential (electrostatic) disorder, delta_mu ~ N(0, s t)
    mu+t      the same, plus bond disorder delta_t ~ N(0, s t) at equal rms

MajoranaChannel.with_disorder applies BOTH at once. That matters: measured
here, bond disorder dominates the loss. At delta_mu_rms = Delta the model
retains 77% under electrostatic disorder alone and 30% when bond disorder is
added at the same rms. Sweeping both while labelling the axis "delta_mu" makes
the channel look far more fragile to electrostatic disorder than it is.

Sub-band occupation and orbital depairing are absent from both modes. Claims
about 2D surface-state backscattering protection cannot be made from this
model at all -- there is no second dimension in it.

Delta_top is read as the third-lowest |E| of the finite chain: the first two
are the hybridised Majorana pair, the third is the bottom of the bulk.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tvqpu.lattice import REV21, MajoranaChannel


def levels(n_sites: int, params, sigma: float, seed: int):
    """(Majorana splitting, bulk edge) for one realisation, in meV.

    Both are needed. ev[2] alone is only a topological gap while a topological
    phase exists; once disorder destroys it, ev[2] is just the lowest localised
    state and the "retention" it implies is meaningless. The splitting ev[0]
    is the diagnostic: it is exponentially small in the topological phase and
    rises to the bulk scale when the phase is gone.
    """
    ch = MajoranaChannel(n_sites=n_sites, params=params)
    if sigma > 0:
        ch = ch.with_disorder(sigma=sigma, seed=seed)
    ev = np.sort(np.abs(ch.spectrum()))
    return float(ev[0]), float(ev[2])


def delta_top(n_sites: int, params, sigma: float, seed: int) -> float:
    return levels(n_sites, params, sigma, seed)[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sites", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--delta", type=float, default=2.0, help="induced gap, meV")
    ap.add_argument("--vz", type=float, default=3.95, help="Zeeman energy, meV")
    a = ap.parse_args()

    p = replace(REV21, mu=0.0, v_z=a.vz, delta=a.delta)
    t = p.t

    print(f"N = {a.n_sites} sites, Delta = {a.delta} meV, V_z = {a.vz} meV, "
          f"t = {t} meV")
    print(f"{a.seeds} disorder realisations per point; median reported\n")

    clean = delta_top(a.n_sites, p, 0.0, 0)
    print(f"clean limit:  Delta_top = {clean:.4f} meV   "
          f"retention = {clean / a.delta * 100:.1f}%\n")

    hdr = (f"{'sigma':>7} {'dmu_rms':>9} {'med Dtop':>9} {'retention':>10} "
           f"{'IQR':>14} {'med split':>11} {'phase':>7}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    sigmas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
              0.075, 0.09, 0.10, 0.12, 0.15, 0.20, 0.30]
    for s in sigmas:
        n = a.seeds if s > 0 else 1
        out = [levels(a.n_sites, p, s, 1000 + k) for k in range(n)]
        spl = np.array([o[0] for o in out])
        bulk = np.array([o[1] for o in out])
        med = float(np.median(bulk))
        med_spl = float(np.median(spl))
        lo, hi = ((float(np.percentile(bulk, 25)), float(np.percentile(bulk, 75)))
                  if n > 1 else (med, med))
        dmu = s * t
        # topological phase is intact while the Majorana pair stays far below
        # the bulk edge; once splitting approaches it, ev[2] is no longer a gap
        intact = med_spl < 0.1 * med
        rows.append((s, dmu, med, med / a.delta, med_spl, intact))
        print(f"{s:7.3f} {dmu:8.3f} {med:9.4f} {med / a.delta * 100:9.1f}% "
              f"{lo:6.3f}-{hi:6.3f} {med_spl:11.2e} "
              f"{'OK' if intact else 'LOST':>7}")

    valid = [r for r in rows if r[5]]
    print()
    if len(valid) < len(rows):
        first_lost = rows[len(valid)]
        print(f"TOPOLOGICAL PHASE LOST at sigma >= {first_lost[0]:.3f} "
              f"(dmu_rms >= {first_lost[1]:.2f} meV): the Majorana splitting is")
        print(f"no longer small against the bulk edge, so retention is "
              f"undefined beyond this point.")
        print(f"Rows past it are reported for completeness and should not be "
              f"read as gaps.")
    print()
    print(f"Within the valid range, retention falls from "
          f"{valid[0][3]*100:.1f}% (clean) to {valid[-1][3]*100:.1f}% at "
          f"dmu_rms = {valid[-1][1]:.2f} meV.")

    # crossing of the architecture's own failure threshold
    fail = [r for r in valid if r[2] < 0.52]
    if fail:
        k = valid.index(fail[0])
        print(f"Crosses the paper's own failure threshold "
              f"(Delta_top < 0.52 meV) between dmu_rms = "
              f"{valid[k-1][1]:.2f} and {fail[0][1]:.2f} meV.")
    else:
        print(f"Stays above the paper's failure threshold (0.52 meV) "
              f"throughout the valid range.")

    print()
    print("NOTE: single-band 1D model. Electrostatic disorder only; no")
    print("sub-band occupation, no orbital depairing. These numbers are an")
    print("UPPER BOUND on retention, not a prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
