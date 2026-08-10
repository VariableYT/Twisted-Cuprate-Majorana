"""Bundle main.tex + exactly the figures it references, into one Overleaf zip.

Reads \\includegraphics from the source rather than globbing figs/, so stray
or superseded PDFs in that directory never end up in the upload.
"""

import re
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
OUT = Path.home() / "Downloads" / "overleaf_upload.zip"

src = (HERE / "main.tex").read_text(encoding="utf-8")
refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src)

missing = [r for r in refs if not (HERE / "figs" / r).exists()]
if missing:
    raise SystemExit(f"referenced but not found in figs/: {missing}")

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(HERE / "main.tex", "main.tex")
    for r in sorted(set(refs)):
        z.write(HERE / "figs" / r, f"figs/{r}")

print(f"wrote {OUT}")
print(f"  main.tex + {len(set(refs))} figures, all referenced:")
for r in sorted(set(refs)):
    print(f"    {r}")
