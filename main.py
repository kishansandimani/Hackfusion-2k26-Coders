import io
import json
import os
import threading
import base64
import uuid
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from rapidfuzz import fuzz, process
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from report import make_pdf

WEIGHTS = "weights/best.pt"
app = FastAPI(title="ChipPal API")

# ---------------- AI models (loaded once) ----------------
model = YOLO(WEIGHTS)
sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics", model_path=WEIGHTS,
    confidence_threshold=0.2, device="cpu")
_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        import easyocr
        _ocr = easyocr.Reader(["en"], gpu=False)
    return _ocr


# ---------------- lookup tables (rule-based, not AI) ----------------
def norm(s):
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


_db = pd.read_csv("components.csv")
_db["key"] = _db["class"].str.lower().str.strip()
INFO = _db.set_index("key").to_dict("index")
_pt = pd.read_csv("parts_db.csv")
_pt["key"] = _pt["part_number"].apply(norm)
PARTS = _pt.set_index("key").to_dict("index")
DEFAULT = {"value_inr": 1.0, "mass_g": 0.3, "hazard_fraction": 0.03,
           "verdict": "Check manually"}
COLORS = {"high": "red", "medium": "orange", "low": "lime"}
JOBS = OrderedDict()
DETECT_LOCK = threading.Lock()  # one detection or OCR job at a time (several users)


def level(h):
    return "high" if h >= 0.10 else "medium" if h >= 0.04 else "low"


# ---------------- detection ----------------
def detect_normal(path, conf):
    res = model.predict(path, conf=conf, imgsz=1024, max_det=1000, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append({"name": res.names[int(b.cls)], "conf": float(b.conf),
                    "box": (x1, y1, x2, y2)})
    return out


def detect_tiled(path, conf, tile):
    sahi_model.confidence_threshold = conf
    result = get_sliced_prediction(
        path, sahi_model, slice_height=tile, slice_width=tile,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
    out = []
    for p in result.object_prediction_list:
        bb = p.bbox
        out.append({"name": p.category.name, "conf": float(p.score.value),
                    "box": (bb.minx, bb.miny, bb.maxx, bb.maxy)})
    return out


# ---------------- chip reading ----------------
def match_part(text):
    clean = norm(text)
    if len(clean) < 4:
        return None, 0
    for key in PARTS:
        if len(key) >= 5 and key in clean:
            return key, 100
    m = process.extractOne(clean, list(PARTS.keys()),
                           scorer=fuzz.partial_ratio, score_cutoff=88)
    if m and len(m[0]) >= 5:
        return m[0], int(m[1])
    return None, 0


def read_chip(orig, box, reader):
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = orig.crop((max(0, x1 - 4), max(0, y1 - 4), x2 + 4, y2 + 4)).convert("L")
    w, h = crop.size
    scale = max(1, 300 // max(1, min(w, h)))
    if scale > 1:
        crop = crop.resize((w * scale, h * scale), Image.LANCZOS)
    best = ("", None, 0)
    for angle in (0, 90, 180, 270):
        arr = np.array(crop.rotate(angle, expand=True))
        txts = reader.readtext(
            arr, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
        text = " ".join(txts)
        part, score = match_part(text)
        if (score, len(text)) > (best[2], len(best[0])):
            best = (text, part, score)
    return best


def build_plan(summary, df, hazard_pct, mass):
    """Salvage plan and score. Rule-based, not AI."""
    total = float(df.value_inr.sum())
    reuse_val = float(df[df.verdict.str.contains("Reuse")].value_inr.sum())
    share = reuse_val / total if total else 0.0
    score = round(60 * share + 40 * (1 - min(hazard_pct / 25, 1)))
    ph = {"care": [], "reuse": [], "recycle": []}
    for _, r in summary.iterrows():
        info = INFO.get(str(r["class"]).lower().strip(), DEFAULT)
        item = {"class": str(r["class"]), "count": int(r["count"]),
                "value": round(float(r["value_inr"]), 1)}
        v = str(r["verdict"])
        if v.startswith("Handle") or level(info["hazard_fraction"]) == "high":
            ph["care"].append(item)
        elif v.startswith("Reuse"):
            ph["reuse"].append(item)
        else:
            ph["recycle"].append(item)
    ph["reuse"].sort(key=lambda x: -x["value"])
    plan = [
        {"id": "care", "title": "1. Remove with care",
         "note": "Do these first, with gloves and ventilation. Do not puncture or crush.",
         "items": ph["care"]},
        {"id": "reuse", "title": "2. De-solder for reuse",
         "note": "Highest estimated value first.", "items": ph["reuse"]},
        {"id": "recycle", "title": "3. Send for material recovery",
         "note": "Low reuse value. Recover the metals through an authorised recycler.",
         "items": ph["recycle"]},
    ]
    return {"plan": plan, "score": int(score), "reuse_share": round(100 * share),
            "haz_g": round(float(df.hazard_mass_g.sum()), 2), "mass_g": round(float(mass), 1)}


# ---------------- API ----------------
def finalize(orig, dets, mode, job_id=None):
    img = orig.copy()
    draw = ImageDraw.Draw(img)
    lw = max(2, max(img.size) // 500)
    rows = []
    for d in dets:
        info = INFO.get(d["name"].lower().strip(), DEFAULT)
        draw.rectangle(d["box"], outline=COLORS[level(info["hazard_fraction"])], width=lw)
        rows.append({"class": d["name"], "confidence": round(d["conf"], 2),
                     "value_inr": info["value_inr"], "mass_g": info["mass_g"],
                     "hazard_mass_g": info["mass_g"] * info["hazard_fraction"],
                     "verdict": info["verdict"]})
    df = pd.DataFrame(rows)
    mass = df.mass_g.sum()
    hazard_pct = 100 * df.hazard_mass_g.sum() / mass if mass else 0
    reusable = int(df.verdict.str.contains("Reuse").sum())
    low = int((df.confidence < 0.4).sum())
    summary = (df.groupby("class")
                 .agg(count=("class", "size"), value_inr=("value_inr", "sum"),
                      verdict=("verdict", "first"))
                 .reset_index().sort_values("count", ascending=False))
    metrics = {"Components detected": len(df), "Reusable parts": reusable,
               "Estimated value (INR)": f"{df.value_inr.sum():.0f}",
               "Estimated hazardous content": f"{hazard_pct:.1f}%",
               "Low-confidence detections": low}

    job_id = job_id or uuid.uuid4().hex[:10]
    W, H = orig.size
    JOBS[job_id] = {"orig": orig, "img": img, "dets": dets, "df": df,
                    "summary": summary, "metrics": metrics, "mode": mode}
    while len(JOBS) > 20:
        JOBS.popitem(last=False)

    shown = img.copy()
    shown.thumbnail((1600, 1600))
    buf = io.BytesIO()
    shown.save(buf, format="JPEG", quality=85)
    return {
        "job_id": job_id,
        "image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
        "classes": sorted(set(model.names.values())),
        "boxes": [{"c": d["name"], "p": round(d["conf"], 3), "m": d["conf"] >= 1.0,
                   "b": [round(d["box"][0] / W, 5), round(d["box"][1] / H, 5),
                         round(d["box"][2] / W, 5), round(d["box"][3] / H, 5)]} for d in dets],
        "mode": mode, "low": low, "ics": sum(1 for d in dets if d["name"].lower() == "ic"),
        "metrics": {"components": len(df), "reusable": reusable,
                    "value": round(float(df.value_inr.sum())),
                    "hazard": round(float(hazard_pct), 1)},
        **build_plan(summary, df, hazard_pct, mass),
        "summary": [{"class": str(r["class"]), "count": int(r["count"]),
                     "value": round(float(r["value_inr"]), 1), "verdict": str(r["verdict"]),
                     "level": level(INFO.get(str(r["class"]).lower().strip(), DEFAULT)["hazard_fraction"])}
                    for _, r in summary.iterrows()],
    }


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), conf: float = Form(0.2),
            tiled: bool = Form(True), tile: int = Form(640)):
    try:
        orig = ImageOps.exif_transpose(
            Image.open(io.BytesIO(file.file.read()))).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")
    path = f"temp_{uuid.uuid4().hex[:8]}.jpg"
    orig.save(path)
    try:
        with DETECT_LOCK:
            dets = detect_tiled(path, conf, tile) if tiled else detect_normal(path, conf)
    finally:
        if os.path.exists(path):
            os.remove(path)
    if not dets:
        raise HTTPException(422, "No components found. Try a lower confidence.")
    return finalize(orig, dets, "tiled" if tiled else "standard")


@app.post("/api/recompute")
def recompute(job_id: str = Form(...), dets: str = Form(...)):
    job = _job(job_id)
    W, H = job["orig"].size
    try:
        items = json.loads(dets)
        new = [{"name": str(i["name"]), "conf": float(i.get("conf", 1.0)),
                "box": (i["box"][0] * W, i["box"][1] * H, i["box"][2] * W, i["box"][3] * H)}
               for i in items]
    except Exception:
        raise HTTPException(400, "Bad edit data.")
    if not new:
        raise HTTPException(422, "No components left. Add one or undo your edits.")
    return finalize(job["orig"], new, job["mode"], job_id)


@app.post("/api/chips")
def chips(job_id: str = Form(...)):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "This analysis expired. Please analyze the photo again.")
    ics = sorted([d for d in job["dets"] if d["name"].lower().strip() == "ic"],
                 key=lambda d: -d["conf"])[:12]
    with DETECT_LOCK:
        reader = get_ocr()
        reads = [read_chip(job["orig"], d["box"], reader) for d in ics]
    out = []
    for i, (text, part, score) in enumerate(reads):
        if part:
            p = PARTS[part]
            out.append({"n": i + 1, "text": text, "part": p["part_number"],
                        "desc": p["description"], "value": float(p["value_inr"]), "match": score})
        else:
            out.append({"n": i + 1, "text": text or "(unreadable)", "part": "Not recognised",
                        "desc": "Generic IC value used",
                        "value": float(INFO.get("ic", DEFAULT)["value_inr"]), "match": 0})
    return {"results": out}


def _job(job_id):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "This analysis expired. Please analyze the photo again.")
    return job


@app.get("/api/report/{job_id}.pdf")
def report_pdf(job_id: str):
    j = _job(job_id)
    pdf = make_pdf(j["img"], j["summary"], j["metrics"], j["mode"])
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="ewaste_report.pdf"'})


@app.get("/api/csv/{job_id}.csv")
def report_csv(job_id: str):
    j = _job(job_id)
    return Response(j["df"].to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="ewaste_report.csv"'})


# The frontend (static/index.html) is served from the same server.
app.mount("/", StaticFiles(directory="static", html=True), name="static")