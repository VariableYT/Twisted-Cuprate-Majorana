"""
robustness_map.py -- find the operating point, not just the design point.

Rev 2.1 fixes (mu, V_z) = (0, 3 meV). Neither was optimised:
  * mu = 0 was chosen as the "sweet spot" where V_z,crit = Delta, and the
    E_F <= 8 meV materials requirement was inherited from the VORTEX mini-gap
    argument -- which does not apply to a gate-defined channel.
  * V_z = 3 meV was chosen as "comfortably inside the topological phase",
    not maximised.

This script maps Delta_top over (mu, V_z) subject to the constraint that
actually binds in a laboratory: the in-plane field.

    V_z = 1/2 g mu_B B,  g = 25  =>  V_z [meV] = 0.7234 * B [T]

so V_z = 3 meV is B = 4.15 T, and a 16 T magnet caps V_z at 11.6 meV. An
unconstrained V_z scan will happily report a huge gap at V_z = 48 meV, which
is 66 T and meaningless. THE FIELD CONSTRAINT IS NOT OPTIONAL.

Also reports what larger mu costs: localisation length xi and the resulting
minimum channel length, since L >= 30 xi sets the die size.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from tvqpu.lattice import REV21, MajoranaChannel

KB = 0.08617333262          # meV/K
G_FACTOR = 25.0
MU_B = 0.05788381806        # meV/T


def vz_to_field(vz: float, g: float = G_FACTOR) -> float:
    """In-plane field required for a given Zeeman energy, in tesla."""
    return vz / (0.5 * g * MU_B)


def field_to_vz(b: float, g: float = G_FACTOR) -> float:
    return 0.5 * g * MU_B * b


def bulk_gap(mu: float, vz: float, delta: float, nk: int = 601) -> float:
    p = replace(REV21, mu=mu, v_z=vz, delta=delta)
    return MajoranaChannel(n_sites=1, params=p).bulk_gap(nk=nk)


def k0_gap(mu: float, vz: float, delta: float) -> float:
    """Closed form for the k = 0 branch."""
    return abs(vz - math.hypot(mu, delta))


def scan(delta: float, b_max: float, mus, nk: int = 601, n_vz: int = 60):
    """Best (V_z, Delta_top) at each mu, subject to B <= b_max."""
    vz_cap = field_to_vz(b_max)
    out = []
    for mu in mus:
        vcrit = math.hypot(mu, delta)
        if vcrit >= vz_cap:
            out.append((mu, None, None, None, "V_z,crit exceeds field limit"))
            continue
        grid = np.linspace(vcrit + 1e-3, vz_cap, n_vz)
        best_g, best_v = -1.0, None
        for vz in grid:
            g = bulk_gap(mu, float(vz), delta, nk=nk)
            if g > best_g:
                best_g, best_v = g, float(vz)
        # is the optimum interior, or pinned at the field cap?
        pinned = abs(best_v - vz_cap) < (grid[1] - grid[0])
        note = "PINNED at field cap" if pinned else "interior optimum"
        out.append((mu, best_v, best_g, vz_to_field(best_v), note))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--b-max", type=float, default=16.0,
                    help="max in-plane field in tesla (9 T and 16 T are the "
                         "common lab superconducting-magnet limits)")
    ap.add_argument("--nk", type=int, default=601)
    args = ap.parse_args()

    print(f"Delta = {args.delta} meV,  field limit {args.b_max} T "
          f"(V_z <= {field_to_vz(args.b_max):.2f} meV)\n")

    print("=== 1. Is the unconstrained optimum physical? ===")
    for vz in (3.0, 6.0, 12.0, 24.0, 48.0):
        g = bulk_gap(0.0, vz, args.delta, nk=args.nk)
        print(f"   V_z = {vz:5.1f} meV -> B = {vz_to_field(vz):6.1f} T, "
              f"Delta_top = {g:6.3f} meV, k=0 form = {k0_gap(0,vz,args.delta):6.3f}")
    print("   (if Delta_top tracks the k=0 form indefinitely, an unbounded")
    print("    V_z scan is measuring the scan range, not the physics)\n")

    print(f"=== 2. Delta_top vs mu, subject to B <= {args.b_max} T ===")
    print(f"{'mu (meV)':>9} {'V_z*':>7} {'B (T)':>7} {'Delta_top':>10} "
          f"{'T_max':>8}  note")
    mus = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0]
    rows = scan(args.delta, args.b_max, mus, nk=args.nk)
    for mu, vz, g, b, note in rows:
        if vz is None:
            print(f"{mu:9.1f} {'--':>7} {'--':>7} {'--':>10} {'--':>8}  {note}")
        else:
            print(f"{mu:9.1f} {vz:7.2f} {b:7.1f} {g:10.3f} "
                  f"{g/(20*KB):7.2f}K  {note}")

    print(f"\n=== 3. What larger mu costs: xi and channel length ===")
    print(f"{'mu (meV)':>9} {'V_z':>6} {'xi (sites)':>11} {'xi (nm)':>9} "
          f"{'L=30xi (um)':>12} {'dE at N=200':>13}")
    for mu, vz, g, b, note in rows:
        if vz is None:
            continue
        ch = MajoranaChannel(n_sites=200,
                             params=replace(REV21, mu=mu, v_z=vz,
                                            delta=args.delta))
        try:
            xi = ch.localization_length()
            split = ch.edge_splitting()
            print(f"{mu:9.1f} {vz:6.2f} {xi:11.1f} {xi*REV21.a_nm:9.0f} "
                  f"{30*xi*REV21.a_nm/1000:12.2f} {split*1e6:12.1f} neV")
        except Exception as e:
            print(f"{mu:9.1f} {vz:6.2f}   xi fit failed: {str(e)[:40]}")

    print(f"\n=== 4. Robustness: Delta_top vs Delta at the chosen point ===")
    print("How much does the gap degrade if the measured pairing is smaller")
    print("than assumed?  Re-optimising V_z at each Delta, within the field cap.\n")
    print(f"{'Delta':>7} {'V_z*':>7} {'B (T)':>7} {'Delta_top':>10} {'T_max':>8}")
    for d in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        vz_cap = field_to_vz(args.b_max)
        vcrit = d
        grid = np.linspace(vcrit + 1e-3, vz_cap, 60)
        best_g, best_v = -1.0, None
        for vz in grid:
            g = bulk_gap(0.0, float(vz), d, nk=args.nk)
            if g > best_g:
                best_g, best_v = g, float(vz)
        print(f"{d:7.2f} {best_v:7.2f} {vz_to_field(best_v):7.1f} "
              f"{best_g:10.3f} {best_g/(20*KB):7.2f}K")

    return 0


if __name__ == "__main__":
    sys.exit(main())
