"""
md_to_pdf.py -- render the project's markdown docs to PDF.

Deliberately small and dependency-light: reportlab only, no pandoc, no LaTeX.
Handles the markdown subset actually used in this repo -- headings, paragraphs,
bold/italic/code spans, links, fenced code blocks, bullet and numbered lists,
pipe tables, blockquotes and horizontal rules.

Not a general markdown implementation. If a doc starts using nested lists or
footnotes, extend it here rather than reaching for a heavyweight toolchain.

Usage:
    python scripts/md_to_pdf.py docs/ARCHITECTURE_PAPER.md
    python scripts/md_to_pdf.py docs/*.md
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, Image, ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#c8c8c8")
CODEBG = colors.HexColor("#f4f4f4")
LINK = colors.HexColor("#1a4f8a")


# --------------------------------------------------------------------------
# Unicode fonts.
#
# THE BUG THIS FIXES: reportlab's built-in Helvetica and Courier are Latin-1
# only. Every Greek letter and math symbol in these documents -- Delta, mu,
# xi, approx, angle brackets, subscripts -- has no glyph in them, and reportlab
# renders a missing glyph as a solid black box. Documents produced before this
# was fixed are full of them.
#
# DejaVu ships with matplotlib and covers the full range, so register it and
# route every style through it. If DejaVu is genuinely absent we fall back,
# but WARN LOUDLY rather than silently producing boxes again.
# --------------------------------------------------------------------------
BODY_FONT, BOLD_FONT, ITALIC_FONT, MONO_FONT = (
    "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier")


def _register_unicode_fonts() -> bool:
    global BODY_FONT, BOLD_FONT, ITALIC_FONT, MONO_FONT
    try:
        import matplotlib
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        from reportlab.pdfbase.ttfonts import TTFont

        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        for name, fn in (("DejaVuSans", "DejaVuSans.ttf"),
                         ("DejaVuSans-Bold", "DejaVuSans-Bold.ttf"),
                         ("DejaVuSans-Oblique", "DejaVuSans-Oblique.ttf"),
                         ("DejaVuSansMono", "DejaVuSansMono.ttf")):
            path = d / fn
            if not path.exists():
                raise FileNotFoundError(path)
            pdfmetrics.registerFont(TTFont(name, str(path)))
        registerFontFamily("DejaVuSans", normal="DejaVuSans",
                           bold="DejaVuSans-Bold",
                           italic="DejaVuSans-Oblique",
                           boldItalic="DejaVuSans-Bold")
        BODY_FONT, BOLD_FONT = "DejaVuSans", "DejaVuSans-Bold"
        ITALIC_FONT, MONO_FONT = "DejaVuSans-Oblique", "DejaVuSansMono"
        return True
    except Exception as exc:                       # pragma: no cover
        print(f"  WARNING: Unicode fonts unavailable ({exc}). Greek and math "
              "symbols WILL render as black boxes.")
        return False


_HAVE_UNICODE = _register_unicode_fonts()


def styles():
    base = getSampleStyleSheet()
    s = {}
    s["body"] = ParagraphStyle(
        "body", parent=base["BodyText"], fontName=BODY_FONT, fontSize=9.5,
        leading=13.5, textColor=INK, alignment=TA_LEFT, spaceAfter=7)
    s["h1"] = ParagraphStyle(
        "h1", parent=s["body"], fontName=BOLD_FONT, fontSize=17,
        leading=21, spaceBefore=4, spaceAfter=10)
    s["h2"] = ParagraphStyle(
        "h2", parent=s["body"], fontName=BOLD_FONT, fontSize=12.5,
        leading=16, spaceBefore=15, spaceAfter=6)
    s["h3"] = ParagraphStyle(
        "h3", parent=s["body"], fontName=BOLD_FONT, fontSize=10.5,
        leading=14, spaceBefore=11, spaceAfter=4)
    s["code"] = ParagraphStyle(
        "code", parent=s["body"], fontName=MONO_FONT, fontSize=8,
        leading=10.5, textColor=INK, leftIndent=8, spaceBefore=4, spaceAfter=8)
    s["quote"] = ParagraphStyle(
        "quote", parent=s["body"], leftIndent=16, rightIndent=10,
        textColor=MUTED, fontName=ITALIC_FONT, borderPadding=4,
        spaceBefore=5, spaceAfter=8)
    s["cell"] = ParagraphStyle(
        "cell", parent=s["body"], fontSize=8, leading=10.5, spaceAfter=0)
    s["cellhead"] = ParagraphStyle(
        "cellhead", parent=s["cell"], fontName=BOLD_FONT)
    s["li"] = ParagraphStyle("li", parent=s["body"], spaceAfter=3)
    s["caption"] = ParagraphStyle(
        "caption", parent=s["body"], fontSize=8, leading=10.5,
        textColor=MUTED, alignment=1, spaceAfter=2)
    return s


def inline(text: str) -> str:
    """Markdown inline spans -> reportlab mini-HTML.

    Order matters: code spans are extracted first and restored last so their
    contents are never interpreted as emphasis.
    """
    stash: list[str] = []

    def keep(m):
        stash.append(html.escape(m.group(1)))
        return f"\x00{len(stash)-1}\x00"

    text = re.sub(r"`([^`]+)`", keep, text)
    text = html.escape(text)
    # links: [label](url) -> label, underlined, colored
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<link href="{m.group(2)}"><font color="#1a4f8a">'
                  f'{m.group(1)}</font></link>', text)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<strike>\1</strike>", text)

    def restore(m):
        return (f'<font face="{MONO_FONT}" size="8.5" backColor="#f0f0f0">'
                f'{stash[int(m.group(1))]}</font>')

    return re.sub(r"\x00(\d+)\x00", restore, text)


def build_table(rows: list[list[str]], s, avail: float):
    header, *body = rows
    ncol = max(len(r) for r in rows)
    data = []
    for i, row in enumerate(rows):
        row = row + [""] * (ncol - len(row))
        st = s["cellhead"] if i == 0 else s["cell"]
        data.append([Paragraph(inline(c), st) for c in row])
    t = Table(data, colWidths=[avail / ncol] * ncol, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ececec")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def convert(md_path: Path, out_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    s = styles()
    doc = SimpleDocTemplate(
        str(out_path), pagesize=LETTER,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
        title=md_path.stem.replace("_", " ").title(), author="Justin Grady")
    avail = LETTER[0] - 1.7 * inch

    flow: list = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # fenced code
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            body = html.escape("\n".join(buf)).replace(" ", "&nbsp;")
            flow.append(Table(
                [[Paragraph(body.replace("\n", "<br/>"), s["code"])]],
                colWidths=[avail], hAlign="LEFT",
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), CODEBG),
                    ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5)])))
            flow.append(Spacer(1, 5))
            continue

        # tables
        if "|" in stripped and i + 1 < n and re.match(
                r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]):
            rows = []
            while i < n and "|" in lines[i]:
                raw = lines[i].strip().strip("|")
                if not re.match(r"^[\s:|-]+$", raw):
                    rows.append([c.strip() for c in raw.split("|")])
                i += 1
            if rows:
                flow.append(build_table(rows, s, avail))
                flow.append(Spacer(1, 8))
            continue

        # lists
        if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+[.)]\s+", line):
            items = []
            numbered = bool(re.match(r"^\s*\d+[.)]\s+", line))
            while i < n and (re.match(r"^\s*[-*+]\s+", lines[i])
                             or re.match(r"^\s*\d+[.)]\s+", lines[i])):
                txt = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", lines[i])
                items.append(ListItem(Paragraph(inline(txt), s["li"]),
                                      leftIndent=16))
                i += 1
            flow.append(ListFlowable(
                items, bulletType="1" if numbered else "bullet",
                start="1" if numbered else None, leftIndent=14,
                bulletFontSize=8))
            flow.append(Spacer(1, 5))
            continue

        # images:  ![caption](path)  on a line of their own
        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
        if m_img:
            cap, src = m_img.group(1), m_img.group(2)
            path = (md_path.parent / src)
            if path.exists():
                try:
                    from reportlab.lib.utils import ImageReader
                    iw, ih = ImageReader(str(path)).getSize()
                    w = min(avail, 4.6 * inch)
                    flow.append(Spacer(1, 4))
                    flow.append(Image(str(path), width=w, height=w * ih / iw))
                    if cap:
                        flow.append(Spacer(1, 2))
                        flow.append(Paragraph(inline(cap), s["caption"]))
                    flow.append(Spacer(1, 8))
                except Exception as exc:
                    flow.append(Paragraph(
                        f"[image failed: {html.escape(src)} - {exc}]", s["body"]))
            else:
                flow.append(Paragraph(
                    f"[missing image: {html.escape(src)}]", s["body"]))
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        if stripped in ("---", "***", "___"):
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.6, color=RULE))
            flow.append(Spacer(1, 6))
            i += 1
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped.lstrip("#").strip()
            flow.append(Paragraph(inline(text), s[f"h{min(level, 3)}"]))
            if level == 1:
                flow.append(HRFlowable(width="100%", thickness=0.8, color=RULE))
                flow.append(Spacer(1, 6))
            i += 1
            continue

        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            flow.append(Paragraph(inline(" ".join(buf)), s["quote"]))
            continue

        # paragraph: gather until blank or a block starter
        start = i
        buf = []
        while i < n and lines[i].strip() and not lines[i].strip().startswith(
                ("#", ">", "```", "---")) and "|" not in lines[i] \
                and not re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", lines[i]):
            buf.append(lines[i].strip())
            i += 1
        if buf:
            flow.append(Paragraph(inline(" ".join(buf)), s["body"]))

        # SAFETY: every branch must consume at least one line. A line holding a
        # stray '|' that is not a well-formed table falls through every branch
        # above, and then the while-condition here fails on its first test --
        # leaving buf empty and i unmoved. That is an infinite loop, and it is
        # exactly what happened the first time this ran. Never remove this.
        if i == start:
            flow.append(Paragraph(inline(stripped), s["body"]))
            i += 1

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(BODY_FONT, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.85 * inch, 0.5 * inch, md_path.stem)
        canvas.drawRightString(LETTER[0] - 0.85 * inch, 0.5 * inch,
                               f"page {doc_.page}")
        canvas.restoreState()

    doc.build(flow, onFirstPage=footer, onLaterPages=footer)


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    for pattern in argv:
        paths = sorted(Path(".").glob(pattern)) if "*" in pattern \
            else [Path(pattern)]
        for p in paths:
            if not p.exists():
                print(f"  missing: {p}")
                continue
            out = p.with_suffix(".pdf")
            convert(p, out)
            print(f"  {p}  ->  {out}  ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
