import io
from datetime import datetime
from fpdf import FPDF


def clean(text):
    return str(text).encode("latin-1", "replace").decode("latin-1")


def make_pdf(img, summary_df, metrics, mode, plan=None, score=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "E-Waste Circuit Analysis Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, clean(f"Generated: {datetime.now():%d %b %Y, %H:%M}   |   Detection mode: {mode}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    im = img.copy()
    im.thumbnail((1400, 1400))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    w, h = im.size
    scale = min(190 / w, 120 / h)
    y0 = pdf.get_y()
    pdf.image(buf, x=10, y=y0, w=w * scale, h=h * scale)
    pdf.set_y(y0 + h * scale + 4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for label, value in metrics.items():
        pdf.cell(80, 7, clean(label))
        pdf.cell(0, 7, clean(value), new_x="LMARGIN", new_y="NEXT")
    if score is not None:
        pdf.cell(80, 7, "Salvage score (own index, 0-100)")
        pdf.cell(0, 7, clean(score), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Component inventory", new_x="LMARGIN", new_y="NEXT")
    widths = [50, 20, 30, 90]
    pdf.set_font("Helvetica", "B", 10)
    for wd, hd in zip(widths, ["Class", "Count", "Value (INR)", "Verdict"]):
        pdf.cell(wd, 7, hd, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 10)
    for _, r in summary_df.iterrows():
        for wd, v in zip(widths, [r["class"], r["count"], f"{r['value_inr']:.1f}", r["verdict"]]):
            pdf.cell(wd, 7, clean(v), border=1)
        pdf.ln()

    if plan:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Salvage plan", new_x="LMARGIN", new_y="NEXT")
        for p in plan:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, clean(p["title"]), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 5, clean(p["note"]))
            pdf.set_font("Helvetica", "", 10)
            if not p["items"]:
                pdf.cell(0, 6, "  Nothing in this group.", new_x="LMARGIN", new_y="NEXT")
            for i in p["items"]:
                pdf.cell(0, 6, clean(f"  {i['class']} x{i['count']}  -  {i['value']} INR"),
                         new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "Value and hazard figures are estimates from a lookup table "
                         "(typical second-hand prices and typical component composition), "
                         "not measurements. The salvage score is a simple index made for this "
                         "project, not a standard. Low-confidence detections should be checked "
                         "by a person.")
    return bytes(pdf.output())