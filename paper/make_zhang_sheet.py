r"""
make_zhang_sheet.py -- Milestone 1 experiment sheet, targeted at an
experimentalist who would actually build the sample.

SCOPE DISCIPLINE. This sheet answers exactly one question: what is the sample,
what is the measurement, and why is this configuration worth testing. Anything
that does not serve that is cut:

  CUT  architecture cross-section, gate/channel top plan  -- Milestone 1 has
       no gates, no channel and no field; including them muddies the ask
  CUT  helical spin-winding detail -- textbook for this audience
  CUT  energy-scale axis -- superseded by the predicted spectra, which is the
       thing an experimentalist actually compares against

  KEPT/ADDED
       test sample elevation + exploded iso (what to build, in what order)
       predicted dI/dV, computed (what the screen should show)
       gap anisotropy: single-layer nodal vs twisted nodeless (the argument)
       prior-art box: the negative result was on the NODAL configuration
       decision thresholds (what the number means)
       the open fabrication question (what is actually being asked)

Provenance: [M] model input/output   [L] literature   [A] assumed
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

RED, GREY = "#c1121f", "0.45"
MONO = "DejaVu Sans Mono"
HV, LT, DM = 1.15, 0.55, 0.6
GT_LS = (0, (3, 2))
CL_LS = (0, (7, 2, 1, 2))

plt.rcParams.update({"font.family": MONO, "mathtext.fontset": "dejavusans"})

fig = plt.figure(figsize=(20, 15))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 200); ax.set_ylim(0, 150)
ax.set_aspect("equal"); ax.axis("off")


#: Global type scale. One knob so the sheet stays legible when printed or
#: viewed at a distance; every block's line spacing below is set to match.
FS = 1.34


def txt(x, y, s, fs=6.0, ha="left", va="center", w="normal", c="k", rot=0):
    ax.text(x, y, s, fontsize=fs * FS, ha=ha, va=va, fontweight=w, color=c,
            rotation=rot, family=MONO)


def line(x0, y0, x1, y1, lw=LT, c="k", ls="-", z=3):
    ax.plot([x0, x1], [y0, y1], color=c, lw=lw, ls=ls, zorder=z,
            solid_capstyle="butt")


def poly(pts, lw=HV, c="k", ls="-", closed=True, z=3):
    p = list(pts) + ([pts[0]] if closed else [])
    ax.plot([q[0] for q in p], [q[1] for q in p], color=c, lw=lw, ls=ls,
            zorder=z, solid_capstyle="butt")


def leader(xt, yt, xl, yl, label, fs=5.4, c="k", ha="left"):
    ax.annotate("", xy=(xt, yt), xytext=(xl, yl),
                arrowprops=dict(arrowstyle="->", color=c, lw=0.55))
    txt(xl + (0.5 if ha == "left" else -0.5), yl, label, fs=fs, ha=ha, c=c)


def ISO(ox, oy, s):
    c30, s30 = 0.8660, 0.5
    def f(x, y, z):
        return ox + s * (x - y) * c30, oy + s * ((x + y) * s30 + z)
    return f


def slab(f, W, D, z0, t, z=3):
    """OPAQUE isometric slab: the three visible faces are filled white so
    nothing shows through.  Wireframe slabs made the exploded stack unreadable
    because every layer was visible through every other one.

    Caller must draw bottom slab first, top slab last (painter's algorithm):
    the viewer is above, so higher slabs occlude lower ones.
    """
    z1 = z0 + t
    faces = [
        [f(0, 0, z1), f(W, 0, z1), f(W, D, z1), f(0, D, z1)],   # top
        [f(0, 0, z0), f(W, 0, z0), f(W, 0, z1), f(0, 0, z1)],   # front, y = 0
        [f(0, 0, z0), f(0, D, z0), f(0, D, z1), f(0, 0, z1)],   # front, x = 0
    ]
    for k, fc in enumerate(faces):
        ax.add_patch(plt.Polygon(fc, closed=True, facecolor="white",
                                 edgecolor="none", zorder=z + k * 0.01))
    poly(faces[0], lw=HV, z=z + 0.05)                            # top outline
    for fc in faces[1:]:
        poly(fc, lw=LT, z=z + 0.05)


# ---- frame -----------------------------------------------------------------
poly([(2, 2), (198, 2), (198, 148), (2, 148)], lw=1.4)
poly([(6, 6), (194, 6), (194, 144), (6, 144)], lw=0.9)
for i in range(6):
    xm = 6 + 188 * (i + 0.5) / 6
    txt(xm, 146.9, f"{i+1}", fs=8, ha="center", w="bold")
    txt(xm, 3.9, f"{i+1}", fs=8, ha="center", w="bold")
for i, ch in enumerate("ABCDE"):
    ym = 144 - 138 * (i + 0.5) / 5
    txt(3.9, ym, ch, fs=8, ha="center", w="bold")
    txt(196.1, ym, ch, fs=8, ha="center", w="bold")
for i in range(1, 6):
    xb = 6 + 188 * i / 6
    line(xb, 144, xb, 148, lw=0.5); line(xb, 2, xb, 6, lw=0.5)
for i in range(1, 5):
    yb = 6 + 138 * i / 5
    line(2, yb, 6, yb, lw=0.5); line(194, yb, 198, yb, lw=0.5)

txt(100, 141.2, "MILESTONE 1 — PROXIMITY-INDUCED GAP AT A TWISTED-CUPRATE / "
    "TOPOLOGICAL-INSULATOR INTERFACE", fs=11.0, ha="center", w="bold")
txt(100, 138.0, "sample definition, measurement, and decision criterion — "
    "one experiment, one number", fs=6.2, ha="center", c=GREY)


# ============================================================================
# 1. TEST SAMPLE — SECTION                              (zone A-B / 1-2)
# ============================================================================
# Stack sits right of centre so the left margin carries dimensions only and
# the interface callout has a clear lane that crosses no geometry.
SX0, SX1 = 27, 59
rows = [(96.0, 102.0, "SUBSTRATE [A]", "≈0.5 mm"),
        (102.0, 105.4, "Bi-2212 BASE, 2 UC", "6.2 nm"),
        (105.4, 108.8, "Bi-2212 45.0° TWIST, 2 UC", "6.2 nm"),
        (108.8, 113.0, "Bi$_2$Se$_3$, 10 QL (EXPOSED)", "9.6 nm")]
poly([(SX0, 96.0), (SX1, 96.0), (SX1, 113.0), (SX0, 113.0)], lw=HV)
for y0, y1, lab, dim in rows:
    if y1 != 113.0:
        line(SX0, y1, SX1, y1, lw=LT)
    txt(SX1 + 7.0, (y0 + y1) / 2, lab, fs=5.4)
    ax.annotate("", xy=(SX0 - 3.0, y0), xytext=(SX0 - 3.0, y1),
                arrowprops=dict(arrowstyle="<->", color="k", lw=0.45))
    txt(SX0 - 4.2, (y0 + y1) / 2, dim, fs=4.9, ha="right")
for xx in np.linspace(SX0 + 4, SX1 - 4, 3):
    line(xx, 96.7, xx + 2.0, 98.1, lw=0.5)

# interface: red line extended a little past the stack on the left, labelled
# there, so the leader never crosses a layer
line(SX0 - 6.5, 108.8, SX1, 108.8, lw=1.5, c=RED, z=5)
txt(SX0 - 7.4, 108.8, "INTERFACE\nUNDER TEST", fs=5.0, ha="right", c=RED)
txt(SX0 - 7.4, 105.0, "Δ INDUCED HERE", fs=4.7, ha="right", c=RED)

# ground return: the substrate is insulating, so tunnelling needs a contact
# to the bilayer. Drawn as a pad on the base layer with a ground symbol.
poly([(SX1, 102.2), (SX1 + 3.2, 102.2), (SX1 + 3.2, 105.2), (SX1, 105.2)],
     lw=0.7)
line(SX1 + 1.6, 102.2, SX1 + 1.6, 100.0, lw=0.7)
for k, hw in enumerate((1.7, 1.1, 0.55)):
    line(SX1 + 1.6 - hw, 100.0 - k * 0.65, SX1 + 1.6 + hw, 100.0 - k * 0.65,
         lw=0.7)
txt(SX1 + 1.6, 103.7, "GND", fs=4.2, ha="center")

poly([(41.4, 122.0), (44.6, 122.0), (43.0, 115.4)], lw=0.9)
txt(45.8, 120.0, "STM / STS TIP", fs=5.6)
ax.annotate("", xy=(43.0, 113.6), xytext=(43.0, 115.1),
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
txt(41.6, 114.6, "dI/dV", fs=5.2, ha="right", c=RED)
txt(45.8, 117.8, "T = 1 K,  B = 0,  BIAS ±4 mV", fs=5.2)

txt(40, 128.4, "1.  TEST SAMPLE — SECTION", fs=7.6, ha="center", w="bold")
txt(40, 126.6, "vertical NTS; no gates, no channel, no applied field",
    fs=5.4, ha="center", c=GREY)
txt(40, 94.8, "STACK BILAYER FIRST, TI LAST → PROXIMITISED SURFACE IS THE "
    "ONE THE TIP SEES", fs=5.2, ha="center", c=GREY)
# Not a detail: every published null on this interface grew the TI in situ;
# the positive report bonded it. Growing it would repeat the failed case.
txt(40, 92.7, "TRANSFER THE Bi$_2$Se$_3$ — DO NOT GROW IT ON THE CUPRATE "
    "(see panel 5)", fs=5.2, ha="center", c=RED, w="bold")


# ============================================================================
# 2. EXPLODED ISOMETRIC — ASSEMBLY ORDER                (zone A-B / 3-4)
# ============================================================================
ISO_OX, ISO_OY, ISO_S = 92, 92, 0.74
f = ISO(ISO_OX, ISO_OY, ISO_S)
W, D = 22, 15
iso_layers = [(0.0, 3.0, "STEP 1 — SUBSTRATE"),
              (7.0, 1.8, "STEP 2 — Bi-2212, BASE"),
              (12.5, 1.8, "STEP 3 — Bi-2212, TWISTED 45.0°"),
              (18.0, 2.1, "STEP 4 — Bi$_2$Se$_3$, 10 QL")]
c0, c1 = f(W / 2, D / 2, -2.5), f(W / 2, D / 2, 23.5)
line(c0[0], c0[1], c1[0], c1[1], lw=0.5, ls=CL_LS, z=1)
# bottom-to-top: the viewer is above, so each slab occludes the one beneath
for i, (z0, t, _) in enumerate(iso_layers):
    slab(f, W, D, z0, t, z=3 + i)

# No twist arc drawn on the slab face: now that the stack is opaque, STEP 4
# correctly occludes most of STEP 3's top surface, so a callout drawn there is
# either hidden or needs a leader across the whole sheet. The twist is called
# out in the step label instead, and panel 4 shows what it does.
for z0, t, lab in iso_layers:
    a = f(W, 0, z0 + t / 2)
    yl = ISO_OY + ISO_S * (11 + z0 + t / 2) + 1.4
    line(a[0], a[1], 114, yl, lw=0.45, z=10)
    txt(114.5, yl, lab, fs=5.3,
        c=RED if "TWISTED" in lab else "k")
txt(114.5, ISO_OY + ISO_S * (11 + 12.5 + 0.9) - 0.9,
    "        45.0° is IN-PLANE (panel 4)", fs=4.7, c=RED)

txt(97, 128.4, "2.  ASSEMBLY ORDER — EXPLODED ISOMETRIC", fs=7.6,
    ha="center", w="bold")
txt(97, 126.6, "opaque, hidden lines removed; NTS; assembled sample has zero gaps",
    fs=5.4, ha="center", c=GREY)


# ============================================================================
# 3. PREDICTED dI/dV — COMPUTED                         (zone A-B / 5-6)
# ============================================================================
PX0, PX1, PY0, PY1 = 137, 192, 98, 124
poly([(PX0, PY0), (PX1, PY0), (PX1, PY1), (PX0, PY1)], lw=0.8)

KB = 0.08617333262


def dynes(E, D, G):
    z = (E - 1j * G) / np.sqrt((E - 1j * G) ** 2 - D ** 2 + 0j)
    return np.abs(np.real(z))


def smear(E, D, G, T):
    w = np.linspace(-6, 6, 1400)
    df = 0.25 / (KB * T) / np.cosh(w / (2 * KB * T)) ** 2
    df /= np.trapezoid(df, w)
    return np.array([np.trapezoid(dynes(e - w, D, G) * df, w) for e in E])


Egrid = np.linspace(-4, 4, 260)
for D, ls_, lab in ((0.5, (0, (1.5, 1.5)), "Δ = 0.5 meV  FAILS"),
                    (1.0, (0, (5, 2)), "Δ = 1.0 meV  VIABLE"),
                    (2.0, "-", "Δ = 2.0 meV  DESIGN PT")):
    g = smear(Egrid, D, 0.05 * D, 1.0)
    xs = PX0 + 3 + (Egrid + 4) / 8 * (PX1 - PX0 - 6)
    ys = PY0 + 3 + np.clip(g, 0, 2.4) / 2.4 * (PY1 - PY0 - 7)
    ax.plot(xs, ys, color="k", lw=1.0 if ls_ == "-" else 0.8, ls=ls_,
            zorder=4)
    txt(PX1 - 2.5, PY1 - 3.0 - 2.2 * (D == 1.0) - 4.4 * (D == 0.5), lab,
        fs=5.0, ha="right")
yn = PY0 + 3 + 1.0 / 2.4 * (PY1 - PY0 - 7)
line(PX0 + 3, yn, PX1 - 3, yn, lw=0.4, ls=CL_LS)
txt(PX1 - 3.5, yn - 1.6, "NORMAL-STATE LEVEL", fs=4.6, c=GREY, ha="right")
for v, lab in ((-4, "−4"), (-2, "−2"), (0, "0"), (2, "2"), (4, "4")):
    xv = PX0 + 3 + (v + 4) / 8 * (PX1 - PX0 - 6)
    line(xv, PY0 + 3, xv, PY0 + 2.0, lw=0.4)
    txt(xv, PY0 + 0.8, lab, fs=4.8, ha="center")
txt((PX0 + PX1) / 2, PY0 - 1.6, "SAMPLE BIAS  V  (mV)", fs=5.2, ha="center")
txt(PX0 - 1.6, (PY0 + PY1) / 2, "dI/dV  (NORMALISED)", fs=5.2, rot=90,
    ha="center")
txt((PX0 + PX1) / 2, 128.4, "3.  PREDICTED SPECTRA — COMPUTED [M]", fs=7.6,
    ha="center", w="bold")
txt((PX0 + PX1) / 2, 126.6, "Dynes-broadened, Γ = 0.05Δ, thermally smeared "
    "at T = 1 K", fs=5.4, ha="center", c=GREY)
txt((PX0 + PX1) / 2, 95.4, "OBSERVABLE: HARD GAP — ZERO-BIAS dI/dV "
    "SUPPRESSED TO A FEW % OF NORMAL", fs=5.2, ha="center")
txt((PX0 + PX1) / 2, 93.4, "A SOFT GAP (≳0.3 OF NORMAL) IS ITSELF AN "
    "INFORMATIVE RESULT", fs=5.0, ha="center", c=GREY)


# ============================================================================
# 4. THE ARGUMENT — NODAL vs NODELESS                   (zone C / 1-3)
# ============================================================================
txt(52, 89.9, "4.  WHY THIS CONFIGURATION — GAP ANISOTROPY |Δ(θ)| ON THE "
    "FERMI SURFACE", fs=7.6, ha="center", w="bold")
txt(52, 88.1, "the published null result was measured on the LEFT-HAND case; "
    "the right-hand case has not been tested", fs=5.4, ha="center", c=GREY)

th = np.linspace(0, 2 * np.pi, 400)


def polar(cx, cy, r, mag, lw=1.0, ls="-"):
    xs = cx + r * mag * np.cos(th)
    ys = cy + r * mag * np.sin(th)
    ax.plot(xs, ys, color="k", lw=lw, ls=ls, zorder=4)


# -- (i) single layer, d-wave: nodes
CX1, CY1, R = 30, 76, 8.4
for a_ in (0, np.pi / 2):
    line(CX1 - R * 1.25 * np.cos(a_), CY1 - R * 1.25 * np.sin(a_),
         CX1 + R * 1.25 * np.cos(a_), CY1 + R * 1.25 * np.sin(a_),
         lw=0.4, ls=CL_LS)
polar(CX1, CY1, R, np.abs(np.cos(2 * th)))
for a_ in (np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4):
    ax.plot(CX1 + R * 1.05 * np.cos(a_), CY1 + R * 1.05 * np.sin(a_),
            "o", ms=4.0, mfc="white", mec=RED, mew=1.0, zorder=6)
line(CX1 - R * .93, CY1 - R * .93, CX1 + R * .93, CY1 + R * .93,
     lw=0.5, c=RED, ls=(0, (4, 2)))
line(CX1 - R * .93, CY1 + R * .93, CX1 + R * .93, CY1 - R * .93,
     lw=0.5, c=RED, ls=(0, (4, 2)))
txt(CX1, CY1 - R - 4.4, "(i)  SINGLE Bi-2212 LAYER", fs=6.0, ha="center",
    w="bold")
txt(CX1, CY1 - R - 6.6, "d$_{x^2-y^2}$ :  |Δ| ∝ |cos 2θ|", fs=5.4,
    ha="center")
txt(CX1, CY1 - R - 8.8, "FOUR NODES — |Δ| = 0 ALONG ⟨110⟩", fs=5.2,
    ha="center", c=RED)

# -- (ii) twisted bilayer, d+id': nodeless
CX2, CY2 = 74, 76
for a_ in (0, np.pi / 2):
    line(CX2 - R * 1.25 * np.cos(a_), CY2 - R * 1.25 * np.sin(a_),
         CX2 + R * 1.25 * np.cos(a_), CY2 + R * 1.25 * np.sin(a_),
         lw=0.4, ls=CL_LS)
polar(CX2, CY2, R, np.ones_like(th))
polar(CX2, CY2, R, np.abs(np.cos(2 * th)), lw=0.5, ls=(0, (2, 2)))
txt(CX2, CY2 - R - 4.4, "(ii)  45° TWISTED BILAYER", fs=6.0, ha="center",
    w="bold")
txt(CX2, CY2 - R - 6.6, "d + i d′ :  |Δ| = CONST", fs=5.4, ha="center")
txt(CX2, CY2 - R - 8.8, "NO NODES — |Δ| > 0 IN EVERY DIRECTION", fs=5.2,
    ha="center", c=RED)
txt(CX2 + R + 1.5, CY2 + R - 1.0, "dashed: single-layer\nlobes, for scale",
    fs=4.6, c=GREY)

ax.annotate("", xy=(CX2 - R - 3.0, CY2), xytext=(CX1 + R + 3.0, CY2),
            arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))
txt((CX1 + CX2) / 2, CY2 + 2.4, "TWIST 45°", fs=5.4, ha="center", w="bold")
txt((CX1 + CX2) / 2, CY2 - 2.4, "REMOVES\nTHE NODES", fs=5.0, ha="center")


# ============================================================================
# 5. PRIOR ART + THE OPEN QUESTION                      (zone C / 4-6)
# ============================================================================
poly([(100, 46), (192, 46), (192, 85), (100, 85)], lw=0.9)
txt(101.6, 83.0, "5.  PRIOR ART ON THIS INTERFACE — AND WHAT IS UNTESTED",
    fs=7.0, w="bold")
line(100, 80.9, 192, 80.9, lw=0.5)
pa = [
 ("E. WANG et al., arXiv:1403.4184 (2014)",
  "Bi$_2$Se$_3$ grown in situ on Bi-2212. NO induced gap observed.",
  "SINGLE LAYER → NODAL. Symmetry argument predicts failure."),
 ("P. WANG et al., Nat. Commun. 3, 1056 (2012)",
  "Mechanically bonded Bi$_2$Se$_3$/Bi-2212. Proximity gap reported to 80 K.",
  "SINGLE LAYER, but bonded rather than grown. Positive."),
 ("RACHMILOWITZ et al., npj Quantum Mater. 5, 72 (2020)",
  "Bi$_2$Te$_3$ on Bi-2212. V-shaped in films → HARD GAP on islands.",
  "COULOMB BLOCKADE, not pairing. A POSITIVE RESULT MUST EXCLUDE IT."),
 ("MICROSOFT QUANTUM, arXiv:2606.03884 (2026)",
  "InAs-Pb tetron: Δ$_{ind}$ 570 µeV → Δ$_T$ 70 µeV, measured at 50 mK.",
  "BENCHMARK: 12% induced-to-topological retention in real hardware."),
]
yy = 79.0
for ref, what, why in pa:
    txt(101.6, yy, ref, fs=5.2, w="bold")
    txt(101.6, yy - 2.4, what, fs=5.0)
    txt(101.6, yy - 4.6, why, fs=5.0, c=RED)
    yy -= 6.6
line(100, yy + 1.4, 192, yy + 1.4, lw=0.5)
txt(101.6, yy - 0.6, "THE 45° TWISTED CONFIGURATION HAS NOT BEEN TESTED.",
    fs=6.2, w="bold")
txt(101.6, yy - 2.9, "The twist converts the pairing to fully gapped d+id′, "
    "so the nodal symmetry", fs=5.2)
txt(101.6, yy - 4.7, "objection that explains the 2014 null result does not "
    "apply to it.", fs=5.2)


# ============================================================================
# 6. DECISION + WHAT IS BEING ASKED                     (zone D-E / 1-3)
# ============================================================================
poly([(8, 33), (96, 33), (96, 57), (8, 57)], lw=0.9)
txt(9.5, 55.2, "6.  DECISION CRITERION — WHAT THE MEASURED NUMBER DECIDES",
    fs=7.0, w="bold")
line(8, 53.6, 96, 53.6, lw=0.5)
hdrs = ("MEASURED Δ", "→ Δ$_{top}$ [M]", "T$_{max}$ [M]", "VERDICT")
xs_ = (10.5, 30.0, 48.0, 64.0)
for xh, h in zip(xs_, hdrs):
    txt(xh, 51.6, h, fs=5.4, w="bold")
line(8, 50.2, 96, 50.2, lw=0.4)
tbl = [("< 0.50 meV", "< 0.52 meV", "< 0.30 K", "ARCHITECTURE FAILS"),
       ("1.00 meV", "1.002 meV", "0.58 K", "VIABLE"),
       ("2.00 meV", "1.944 meV", "1.13 K", "DESIGN POINT"),
       ("3.00 meV", "2.825 meV", "1.64 K", "COMFORTABLE")]
yy = 48.2
for row in tbl:
    for xh, cell in zip(xs_, row):
        txt(xh, yy, cell, fs=5.2,
            c=RED if cell == "ARCHITECTURE FAILS" else "k")
    yy -= 2.75
line(8, yy + 1.3, 96, yy + 1.3, lw=0.5)
txt(9.5, yy - 0.9, "EVERY DOWNSTREAM NUMBER IN THE ARCHITECTURE SCALES FROM "
    "THIS ONE MEASUREMENT.", fs=5.2, w="bold")
# The Delta -> Delta_top conversion is a second, separate risk: the table above
# assumes the clean-limit value. Stating the bound is more useful to a
# fabricator than quoting 97% unqualified.
txt(9.5, yy - 2.8, "Δ$_{top}$/Δ ABOVE IS THE CLEAN LIMIT (0.97); REAL "
    "DEVICES REACH 0.12 (panel 5). WE CROSS THE THRESHOLD NEAR "
    "δ$_{rms}$ ≈ Δ — MANUSCRIPT SEC. VI.", fs=5.0)


# ============================================================================
# 7. CONTROLS, SAMPLE REQUIREMENTS, OPEN QUESTIONS      (zone D-E / 1-3)
# ============================================================================
# Panel 5 raises the Coulomb-blockade objection; this panel answers it. Raising
# a trap without showing the way out reads worse than not raising it.
poly([(8, 7.0), (96, 7.0), (96, 31.5), (8, 31.5)], lw=0.9)
txt(9.5, 29.8, "7.  WHAT WOULD MAKE A RESULT CONVINCING", fs=7.0, w="bold")
line(8, 28.2, 96, 28.2, lw=0.5)

# lead-in bold, continuation indented -- inline format so the whole protocol
# fits without dropping any of it
lines7 = [
 ("CONTROL 1 — TEMPERATURE.", RED,
  " A proximity gap closes at the cuprate T$_c$ (~90 K);"),
 (None, None, "a Coulomb-blockade gap does not. dI/dV vs T is the discriminator."),
 ("CONTROL 2 — UNTWISTED REFERENCE.", RED,
  " A 0° stack from the same run should"),
 (None, None, "show the nodal V-shape of panel 4(i); the 45° stack a hard gap."),
 ("CONTROL 3 — ISLAND SIZE.", RED,
  " A charging gap scales with inverse capacitance"),
 (None, None, "and varies with lateral extent; a proximity gap does not (ref. above)."),
 ("SAMPLE.", "k",
  " Ground contact to the bilayer required; tens of μm lateral extent [A]."),
 ("OPEN — FOR YOUR VIEW.", RED,
  " Angular tolerance on the twist is not computed,"),
 (None, None, "and whether cryogenic two-cuprate assembly transfers to a cuprate/TI"),
 (None, None, "interface cannot be settled by simulation."),
]
yy = 26.2
for head, col, rest in lines7:
    if head is None:
        txt(11.2, yy, rest, fs=5.0)
    else:
        txt(9.5, yy, head, fs=5.0, w="bold", c=col)
        txt(9.5 + 0.62 * len(head), yy, rest, fs=5.0)
    yy -= 1.79
txt(9.5, yy - 0.2, "A NULL RESULT DECIDES THE ARCHITECTURE AS FIRMLY AS A "
    "POSITIVE ONE.", fs=5.2, w="bold")


# ============================================================================
# NOTES / TITLE BLOCK                                   (zone D-E / 4-6)
# ============================================================================
poly([(100, 25), (192, 25), (192, 44), (100, 44)], lw=0.9)
txt(101.6, 42.2, "NOTES:", fs=6.2, w="bold")
notes = [
 "1. PROVENANCE: [M] COMPUTED FROM THE VALIDATED BdG MODEL",
 "   (tvqpu.lattice, 17-CHECK GATE); [L] LITERATURE; [A] ASSUMED.",
 "2. ALL LAYER THICKNESSES ARE [A]. THE MEASUREMENT REQUIRES ONLY",
 "   A PROXIMITISED TI SURFACE ACCESSIBLE TO A PROBE.",
 "3. PANEL 3 USES THE SAME MODEL AS THE DECISION TABLE.",
 "4. SECTION AND ISOMETRIC ARE VERTICALLY NTS.",
 "5. GRADY 2026 — github.com/VariableYT/Twisted-Cuprate-Majorana",
]
yy = 39.8
for ln in notes:
    txt(101.6, yy, ln, fs=4.9)
    yy -= 2.15

TB0, TB1 = 100, 192
poly([(TB0, 8), (TB1, 8), (TB1, 25), (TB0, 25)], lw=1.1)
line(TB0 + 20, 8, TB0 + 20, 25, lw=0.6)
txt(TB0 + 10, 20.6, "VS", fs=10.5, ha="center", w="bold")
ax.add_patch(plt.Circle((TB0 + 10, 20.6), 3.0, fill=False, lw=1.0))
txt(TB0 + 10, 15.2, "VARIABLE SYSTEMS", fs=5.4, ha="center", w="bold")
txt(TB0 + 10, 13.2, "EL DORADO HILLS, CA", fs=4.4, ha="center")
txt(TB0 + 10, 9.6, "GEOMETRY SHEET — NOT FABRICATION DATA", fs=4.0,
    ha="center")
rows_tb = [
    ("PROJECT:", "TWISTED-CUPRATE MAJORANA CHANNEL — MILESTONE 1"),
    ("SUBJECT:", "PROXIMITY GAP AT Bi-2212(45°) / Bi$_2$Se$_3$ INTERFACE"),
    ("DRAWING:", "SECTION / ISO / SPECTRA / ANISOTROPY / DECISION"),
]
yy = 23.2
for k, v in rows_tb:
    txt(TB0 + 21.5, yy, k, fs=4.8, w="bold")
    txt(TB0 + 33.0, yy, v, fs=4.8)
    line(TB0 + 20, yy - 1.4, TB1, yy - 1.4, lw=0.4)
    yy -= 3.15
txt(TB0 + 21.5, yy, "AUTHOR", fs=4.3, w="bold")
txt(TB0 + 21.5, yy - 2.2, "J. I. GRADY", fs=4.5)
txt(TB0 + 42.0, yy, "DATE / SCALE", fs=4.3, w="bold")
txt(TB0 + 42.0, yy - 2.2, "2026-08-07   NTS", fs=4.5)
txt(TB0 + 62.0, yy, "DOC NO", fs=4.3, w="bold")
txt(TB0 + 62.0, yy - 2.2, "VS-TCM-M1-REV1", fs=4.5)
for xv in (TB0 + 41.0, TB0 + 61.0):
    line(xv, 8, xv, yy + 1.1, lw=0.4)

fig.savefig(OUT / "milestone1_sheet.png", dpi=200, facecolor="white")
fig.savefig(OUT / "milestone1_sheet.pdf", facecolor="white")
plt.close(fig)
print("  milestone1_sheet.png / .pdf")
