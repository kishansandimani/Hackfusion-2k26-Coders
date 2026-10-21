"""PDF report builder with no extra libraries (only Pillow, which the app already uses).
Writes a plain multi-page A4 PDF: banner, photo, headline numbers, parts table, salvage plan."""
import io
import textwrap
from datetime import datetime

from PIL import Image

PW, PH, M = 595, 842, 40  # A4 in points, margin
COL = {"ink": "#171b2b", "soft": "#5d6178", "pink": "#ff6584", "lav": "#ddd3ff", "mint": "#c8f3dc",
       "gold": "#ffe69a", "rose": "#ffd0de", "line": "#d9d6ee", "zebra": "#f7f5ff", "white": "#ffffff"}


def clean(text):
    return str(text).replace("\u20b9", "Rs. ").encode("cp1252", "replace").decode("cp1252")


def _rgb(h):
    h = h.lstrip("#")
    return " ".join(f"{int(h[i:i + 2], 16) / 255:.3f}" for i in (0, 2, 4))


def _esc(text):
    return clean(text).encode("cp1252").replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _wrap(text, size, width, bold=False):
    per = max(8, int(width / (size * (0.56 if bold else 0.5))))
    return textwrap.wrap(clean(text), per) or [""]


class _Doc:
    def __init__(self):
        self.pages, self.cur, self.y, self.jpeg, self.jsize = [], None, 0, None, (0, 0)
        self.new_page()

    def new_page(self):
        self.cur = []
        self.pages.append(self.cur)
        self.y = PH - M

    def need(self, h):
        if self.y - h < M + 24:
            self.new_page()

    def rect(self, x, y, w, h, fill=None, stroke=None, lw=0.7):
        ops = []
        if fill:
            ops.append(f"{_rgb(fill)} rg")
        if stroke:
            ops.append(f"{_rgb(stroke)} RG {lw} w")
        ops.append(f"{x:.1f} {y:.1f} {w:.1f} {h:.1f} re " + ("B" if fill and stroke else "f" if fill else "S"))
        self.cur.append(" ".join(ops).encode())

    def text(self, s, x, y, size=11, font="F1", color="ink"):
        self.cur.append(f"BT /{font} {size} Tf {_rgb(COL[color])} rg {x:.1f} {y:.1f} Td (".encode() + _esc(s) + b") Tj ET")

    def para(self, s, size=10, font="F1", color="soft", indent=0, lead=1.35):
        bold = font == "F2"
        for ln in _wrap(s, size, PW - 2 * M - indent, bold):
            self.need(size * lead)
            self.y -= size * lead
            self.text(ln, M + indent, self.y, size, font, color)

    def heading(self, s):
        self.need(40)
        self.y -= 26
        self.rect(M, self.y - 2, 4, 16, fill=COL["pink"])
        self.text(s, M + 12, self.y, 14, "F2")
        self.y -= 6

    def build(self):
        n = len(self.pages)
        for i, pg in enumerate(self.pages):
            pg.append(f"BT /F1 8 Tf {_rgb(COL['soft'])} rg {PW / 2 - 24:.1f} 24 Td (Page {i + 1} of {n}) Tj ET".encode())
        out, offs = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"), {}
        img_id = 6 if self.jpeg else None
        first = 7 if self.jpeg else 6
        kids = " ".join(f"{first + 2 * i} 0 R" for i in range(n))
        objs = {1: b"<< /Type /Catalog /Pages 2 0 R >>", 2: f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
                3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
                4: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
                5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>"}
        if self.jpeg:
            w, h = self.jsize
            objs[6] = (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB "
                       f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(self.jpeg)} >>\nstream\n").encode() + self.jpeg + b"\nendstream"
        for i, pg in enumerate(self.pages):
            pid, cid = first + 2 * i, first + 2 * i + 1
            xo = f" /XObject << /Im1 {img_id} 0 R >>" if self.jpeg and i == 0 else ""
            objs[pid] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PW} {PH}] /Contents {cid} 0 R "
                         f"/Resources << /Font << /F1 3 0 R /F2 4 0 R /F3 5 0 R >>{xo} >> >>").encode()
            body = b"\n".join(pg)
            objs[cid] = f"<< /Length {len(body)} >>\nstream\n".encode() + body + b"\nendstream"
        for k in sorted(objs):
            offs[k] = len(out)
            out += f"{k} 0 obj\n".encode() + objs[k] + b"\nendobj\n"
        x = len(out)
        out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
        for k in sorted(objs):
            out += f"{offs[k]:010d} 00000 n \n".encode()
        out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF".encode()
        return bytes(out)


def make_pdf(img, summary_df, metrics, mode, plan=None, score=None):
    d = _Doc()
    # banner
    d.rect(0, PH - 78, PW, 78, fill=COL["pink"])
    d.text("E-Waste Circuit Analysis Report", M, PH - 40, 20, "F2", "white")
    d.text(f"Generated {datetime.now():%d %b %Y, %H:%M}   |   Detection mode: {mode}", M, PH - 60, 9.5, "F1", "white")
    d.y = PH - 78 - 14
    # photo
    im = img.convert("RGB")
    im.thumbnail((1400, 1400))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    d.jpeg, d.jsize = buf.getvalue(), im.size
    w, h = im.size
    k = min((PW - 2 * M) / w, 290 / h)
    dw, dh = w * k, h * k
    d.y -= dh
    d.rect(M - 2, d.y - 2, dw + 4, dh + 4, stroke=COL["line"], lw=1.2)
    d.cur.append(f"q {dw:.1f} 0 0 {dh:.1f} {M:.1f} {d.y:.1f} cm /Im1 Do Q".encode())
    d.y -= 10
    # headline numbers
    d.heading("Summary")
    keys = [("Components detected", "Parts found", "lav"), ("Reusable parts", "Reusable", "mint"),
            ("Estimated value (INR)", "Value (INR)", "gold"), ("Estimated hazardous content", "Hazardous", "rose")]
    bw, gap = (PW - 2 * M - 3 * 10) / 4, 10
    d.need(70)
    top = d.y - 6
    for i, (key, label, c) in enumerate(keys):
        x = M + i * (bw + gap)
        d.rect(x, top - 52, bw, 52, fill=COL[c])
        d.text(str(metrics.get(key, "-")), x + 10, top - 28, 20, "F2")
        d.text(label, x + 10, top - 44, 8.5, "F1", "soft")
    d.y = top - 62
    extra = [(k, v) for k, v in metrics.items() if k not in {a for a, _, _ in keys}]
    if score is not None:
        extra.append(("Salvage score (our own index, 0 to 100)", score))
    for k_, v_ in extra:
        d.para(f"{k_}: {v_}", 10, "F1", "ink")
    # table
    d.heading("Component inventory")
    cols = [("Class", 150), ("Count", 50), ("Value (INR)", 80), ("Verdict", 235)]

    def row(vals, fill=None, bold=False):
        d.need(18)
        d.y -= 16
        if fill:
            d.rect(M, d.y - 4, sum(c for _, c in cols), 16, fill=fill)
        x = M
        for (name, cw), v in zip(cols, vals):
            s = clean(v)
            maxc = max(4, int((cw - 8) / (9.5 * (0.56 if bold else 0.5))))
            d.text(s if len(s) <= maxc else s[:maxc - 1] + ".", x + 4, d.y, 9.5, "F2" if bold else "F1", "white" if bold and fill == COL["ink"] else "ink")
            x += cw
    row([c for c, _ in cols], fill=COL["ink"], bold=True)
    for i, (_, r) in enumerate(summary_df.iterrows()):
        row([r["class"], r["count"], f"{float(r['value_inr']):.1f}", r["verdict"]], fill=COL["zebra"] if i % 2 else None)
    d.y -= 6
    # salvage plan
    if plan:
        d.heading("Salvage plan")
        tint = {"care": "rose", "reuse": "mint", "recycle": "lav"}
        for p in plan:
            d.need(70)
            d.y -= 22
            d.rect(M, d.y - 4, PW - 2 * M, 18, fill=COL[tint.get(p["id"], "lav")])
            d.text(p["title"], M + 8, d.y, 11, "F2")
            d.y -= 2
            d.para(p["note"], 9, "F3", "soft", indent=8)
            if not p["items"]:
                d.para("Nothing in this group.", 9.5, "F1", "ink", indent=8)
            for it in p["items"]:
                d.para(f"{it['class']}  x{it['count']}   -   {it['value']} INR", 9.5, "F1", "ink", indent=8)
    # disclaimer
    d.y -= 10
    d.para("Value and hazard figures are estimates from a lookup table (typical second-hand prices and typical "
           "component composition), not measurements. The salvage score is a simple index made for this project, "
           "not a standard. Low-confidence detections should be checked by a person.", 8.5, "F3", "soft")
    return d.build()