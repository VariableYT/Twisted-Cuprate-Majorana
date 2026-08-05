"""
sigma_tolerance_sweep.py -- derive the electrostatic disorder tolerance.

Converts Rev 2.1's sigma <= 0.2% from an ASSERTED spec into a computed one,
by locating the disorder strength at which the topological order parameter
collapses, with finite-size scaling.

WHY THE EARLIER ATTEMPT DID NOT SETTLE IT (ARCHITECTURE.md section 2.5):
    L = 60, 8 realizations, W <= 4.  The order parameter never collapsed --
    still strongly ordered at W = 4, twice the clean phase boundary -- so
    there was no threshold to find.  Eight realizations also leaves the
    standard deviation itself uncertain at ~25%.

WHAT THIS RUN DOES DIFFERENTLY:
    W to 10, 50 realizations per point, L = 40/60/80/120 for finite-size
    scaling.  Resumable via an append-only ledger, because it is a 3-5 hour
    job on a laptop that closes.

UNITS -- READ tvqpu.dmrg.tolerance_in_gap_units BEFORE QUOTING ANYTHING.
    This chain is dimensionless.  A threshold expressed as W/t does NOT
    transfer to the device, because the toy chain runs at gap/t ~ 1 while
    Rev 2.1 runs at 1.05/20 = 0.05.  Only the ratio delta-mu_rms / gap
    transfers.  Rev 2.1 reference points: spec sigma = 0.2% -> ratio 0.038;
    observed onset sigma ~ 3% -> ratio 0.571.

Usage:
    python scripts/sigma_tolerance_sweep.py --v-int 0.0 --out runs/sigma_v0
    python scripts/sigma_tolerance_sweep.py --report --out runs/sigma_v0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from tvqpu.dmrg import (  # noqa: E402
    clean_gap, disorder_threshold, ground_state, tolerance_in_gap_units,
    EnsembleResult,
)
from tvqpu.lattice import InteractingChain  # noqa: E402

W_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
L_VALUES = (40, 60, 80, 120)

#: Collapse criteria, as fractions of the clean order parameter.  95% is
#: "degradation has begun", which is the criterion Rev 2.1's sigma ~ 3% onset
#: actually uses; 50% is a full collapse.  Quoting only the 50% number against
#: their onset would overstate the agreement by several x.
CRITERIA = (0.95, 0.90, 0.75, 0.50)


def ledger_path(out: Path) -> Path:
    return out / "ledger.jsonl"


def load_done(out: Path) -> set[tuple]:
    p = ledger_path(out)
    if not p.exists():
        return set()
    done = set()
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # a torn final line from a hard kill
            done.add((r["L"], r["w"], r["seed"], r["v_int"]))
    return done


def append(out: Path, record: dict) -> None:
    with ledger_path(out).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())     # survive a laptop lid, not just a clean exit


def run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    print(f"resuming: {len(done)} points already in the ledger")

    w_values = [w for w in W_VALUES
                if (w == 0.0 or w >= args.w_min) and w <= args.w_max]
    todo = [(L, w, s) for L in L_VALUES for w in w_values
            for s in range(args.realizations)
            if (L, w, s, args.v_int) not in done
            and not (w == 0.0 and s > 0)]      # clean case has one realization
    print(f"{len(todo)} points to run")
    t0 = time.time()

    for i, (L, w, seed) in enumerate(todo):
        chain = InteractingChain(n_sites=L, t=1.0, delta=1.0, mu=args.mu,
                                 v_int=args.v_int)
        if w > 0.0:
            chain = chain.with_disorder(w=w, seed=seed)
        t = time.time()
        r = ground_state(chain, chi=args.chi, max_sweeps=args.max_sweeps)
        append(out, {
            "L": L, "w": w, "seed": seed, "v_int": args.v_int, "mu": args.mu,
            "chi": args.chi, "energy": r.energy,
            "order_parameter": r.order_parameter, "entropy": r.entropy,
            "bond_dim": r.bond_dim, "bond_saturated": r.bond_saturated,
            "energy_delta": r.energy_delta, "converged": r.converged,
            "trustworthy": r.trustworthy, "seconds": time.time() - t,
        })
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(todo) - i - 1) / max(rate, 1e-9) / 60
            print(f"  {i+1}/{len(todo)}  L={L} W={w} seed={seed}  "
                  f"<YY>={r.order_parameter:+.4f}  "
                  f"chi={r.bond_dim}{'(SAT)' if r.bond_saturated else ''}  "
                  f"ETA {eta:.0f} min", flush=True)

    print(f"done in {(time.time()-t0)/60:.1f} min")
    return report(args)


def report(args) -> int:
    out = Path(args.out)
    p = ledger_path(out)
    if not p.exists():
        print("no ledger yet")
        return 1
    rows = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows = [r for r in rows if r["v_int"] == args.v_int]
    if not rows:
        print("no rows for this V")
        return 1

    gap = clean_gap(InteractingChain(t=1.0, delta=1.0, mu=args.mu))
    print(f"\nmu = {args.mu}, V = {args.v_int}, clean gap = {gap:.3f} t")
    print(f"Rev 2.1 reference: spec ratio 0.038, observed onset ratio 0.571\n")

    saturated = sum(r["bond_saturated"] for r in rows)
    if saturated:
        print(f"WARNING: {saturated}/{len(rows)} points saturated the bond "
              f"dimension -- their truncation error is unknown. See "
              f"DMRGResult.bond_saturated.\n")

    print(f"{'L':>5} {'W':>6} {'<|YY|>':>9} {'sd':>8} {'sem':>8} {'n':>4}")
    per_L: dict[int, list] = {}
    for L in sorted({r["L"] for r in rows}):
        ens = []
        for w in sorted({r["w"] for r in rows if r["L"] == L}):
            vals = [abs(r["order_parameter"]) for r in rows
                    if r["L"] == L and r["w"] == w]
            if not vals:
                continue
            a = np.asarray(vals)
            sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
            ens.append(EnsembleResult(w=w, mean=float(a.mean()), std=sd,
                                      sem=sd / max(np.sqrt(len(a)), 1),
                                      n_realizations=len(a),
                                      n_trustworthy=len(a)))
            print(f"{L:5d} {w:6.2f} {a.mean():9.4f} {sd:8.4f} "
                  f"{sd/max(np.sqrt(len(a)),1):8.4f} {len(a):4d}")
        per_L[L] = ens
        print()

    # The threshold depends strongly on the collapse criterion, and Rev 2.1's
    # "degradation onset at sigma ~ 3%" is where degradation BEGINS -- not
    # where the order parameter has halved.  Comparing a 50% criterion to
    # their onset would overstate the agreement, so report several.
    print(f"{'L':>5} " + " ".join(f"{'W_c@'+str(int(f*100))+'%':>11}"
                                  for f in CRITERIA))
    for L, ens in per_L.items():
        cells = []
        for f in CRITERIA:
            wc = disorder_threshold(ens, fraction=f)
            cells.append("       none" if wc is None else f"{wc:11.3f}")
        print(f"{L:5d} " + " ".join(cells))

    print(f"\n{'L':>5} " + " ".join(f"{'ratio@'+str(int(f*100))+'%':>12}"
                                    for f in CRITERIA)
          + "     [Rev 2.1: onset 0.571, spec 0.038]")
    for L, ens in per_L.items():
        cells = []
        for f in CRITERIA:
            wc = disorder_threshold(ens, fraction=f)
            cells.append("        none" if wc is None
                         else f"{tolerance_in_gap_units(wc, gap):12.3f}")
        print(f"{L:5d} " + " ".join(cells))

    # Per-criterion extrapolation.  The 50% row alone (what a single naive
    # np.polyfit would give) is the WRONG number to compare against Rev 2.1's
    # onset -- see the CRITERIA comment above.  Report all four, and flag
    # non-monotonicity explicitly rather than silently trusting a linear fit
    # through points that do not form a trend: with n=50 the sem on <|YY|> is
    # ~0.03-0.04, which is enough to make W_c(L) jump around by more than the
    # finite-size trend itself. Never round this into one "the" number.
    print(f"\n{'criterion':>10} "
          + " ".join(f"{'L='+str(L):>8}" for L in sorted(per_L))
          + f" {'extrap':>8} {'ratio':>7}  monotonic?")
    for f in CRITERIA:
        wcs = {L: disorder_threshold(ens, fraction=f) for L, ens in per_L.items()}
        if any(w is None for w in wcs.values()):
            print(f"{int(f*100):>9}%  -- sweep did not reach this threshold "
                  "at every L --")
            continue
        Ls = np.array(sorted(wcs))
        Ws = np.array([wcs[L] for L in Ls])
        diffs = np.diff(Ws)
        mono = bool(np.all(diffs <= 1e-9) or np.all(diffs >= -1e-9))
        slope, intercept = np.polyfit(1.0 / Ls, Ws, 1)
        ratio = tolerance_in_gap_units(intercept, gap)
        print(f"{int(f*100):>9}% " + " ".join(f"{w:8.3f}" for w in Ws)
              + f" {intercept:8.3f} {ratio:7.3f}  "
              + ("yes" if mono else "NO -- fit is through scatter, not a trend"))
    print("\nIf any row above says NOT monotonic, do not quote its extrapolated\n"
          "value as converged. Increase --realizations (sem ~ 1/sqrt(n)) before\n"
          "trusting the fit -- see the default's docstring.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/sigma")
    p.add_argument("--v-int", type=float, default=0.0)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--chi", type=int, default=96)
    p.add_argument("--realizations", type=int, default=200,
                   help="SEM ~ 1/sqrt(n); the completed sigma_v0 baseline used "
                        "n=50, which gave sem ~0.03-0.04 on the order parameter "
                        "and non-monotonic W_c vs L purely from that noise. "
                        "200 cuts sem roughly in half.")
    p.add_argument("--max-sweeps", type=int, default=25)
    p.add_argument("--w-min", type=float, default=0.0,
                   help="skip disorder strengths below this (W=0 always kept)")
    p.add_argument("--w-max", type=float, default=1e9,
                   help="skip disorder strengths above this")
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    return report(args) if args.report else run(args)


if __name__ == "__main__":
    sys.exit(main())
