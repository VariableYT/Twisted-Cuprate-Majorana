# Paper source

`main.tex` — REVTeX 4-2 (APS/PRB style), compiles with pdfLaTeX on Overleaf.

## Build

1. Upload this whole `paper/` directory to Overleaf (including `figs/`).
2. Set the compiler to **pdfLaTeX** and the main document to `main.tex`.
3. Compile. No custom packages beyond `revtex4-2`, `graphicx`, `amsmath`,
   `amssymb`, `bm`, `hyperref`, `xcolor` — all present in Overleaf's default
   TeX Live.

## Regenerating figures

    python paper/make_figures.py

Writes six vector PDFs into `figs/`. Every figure is computed from
`tvqpu.lattice`, the same BdG model the test suite validates — none are
illustrative.

| figure | content |
|---|---|
| `fig1_spectrum` | BdG spectrum vs V_z; the topological transition |
| `fig2_branches` | **key figure** — the two competing minima and the optimum |
| `fig3_map` | Δ_top over (µ, V_z) with field ceilings |
| `fig4_robustness` | Δ_top vs Δ, fixed vs optimised V_z |
| `fig5_localisation` | Majorana density at both operating points |
| `fig6_tunneling` | predicted Milestone 1 tunneling spectra |

## PAPER_PREVIEW.pdf

`PAPER_PREVIEW.md/.pdf` is a **rendered preview, not the manuscript** — the
same argument and the same figures, laid out single-column by
`scripts/md_to_pdf.py` so it can be read without a TeX toolchain. Equation
layout, references and column breaks all differ from `main.tex`. Compile
`main.tex` on Overleaf for the real thing.
