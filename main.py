import hashlib
import io
import json
import os
import re
import threading
import traceback
import time
import base64
import uuid
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from rapidfuzz import fuzz, process
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO

from auth import (COOKIE, current_user, delete_user, is_guest, optional_user,
                  router as auth_router, verify_password)
from compliance import audit
from enhance import enhance_if_needed
from repair import equiv_for, guides_for
from report import make_pdf
from scores import award, delete as delete_scores, profile as score_profile, router as scores_router
import history

WEIGHTS = "weights/best.pt"
app = FastAPI(title="ChipPal API")
app.include_router(auth_router)
app.include_router(scores_router)

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
for _k, _v in {
    "microcontroller": {"value_inr": 40.0, "mass_g": 1.0, "hazard_fraction": 0.05,
                        "verdict": "Reuse / recover metals"},
    "transistor_diode": {"value_inr": 1.2, "mass_g": 0.25, "hazard_fraction": 0.03,
                         "verdict": "Reuse"},
}.items():
    INFO.setdefault(_k, _v)


def _fill(v):
    """Make sure every class has a value range and a source note."""
    val = float(v["value_inr"])
    lo, hi, src = v.get("value_low"), v.get("value_high"), v.get("source")
    v["value_low"] = float(lo) if lo is not None and not pd.isna(lo) else round(val * 0.6, 2)
    v["value_high"] = float(hi) if hi is not None and not pd.isna(hi) else round(val * 1.5, 2)
    v["source"] = str(src) if src is not None and not pd.isna(src) else "rough estimate, not verified"
    return v


INFO = {k: _fill(dict(v)) for k, v in INFO.items()}
DEFAULT = _fill(DEFAULT)
COLORS = {"high": "red", "medium": "orange", "low": "lime"}
JOBS = OrderedDict()
DETECT_LOCK = threading.Lock()  # one detection or OCR job at a time (several users)
PROGRESS = OrderedDict()        # progress id -> {"pct", "stage", "owner", "ts"}: how far an analysis has got
PID_RE = re.compile(r"^[A-Za-z0-9_-]{6,40}$")


def _prog(pid, owner, pct, stage):
    """Report analysis progress. The page polls /api/progress/<id> and shows a percentage."""
    if not pid or not PID_RE.match(pid):
        return
    now = time.time()
    for k in [k for k, v in PROGRESS.items() if now - v["ts"] > 600]:
        PROGRESS.pop(k, None)
    PROGRESS[pid] = {"pct": round(min(100, max(0, pct))), "stage": stage, "owner": owner, "ts": now}


def level(h):
    return "high" if h >= 0.10 else "medium" if h >= 0.04 else "low"


def purge_old(max_age=3600):
    """Forget analyses (and their photos) after an hour."""
    now = time.time()
    for k in [k for k, v in JOBS.items() if now - v.get("ts", now) > max_age]:
        JOBS.pop(k, None)


def quality_tips(orig):
    """Friendly photo advice. Simple heuristics, not a guarantee."""
    tips = []
    w, h = orig.size
    if min(w, h) < 800:
        tips.append(f"Low resolution ({w} x {h} px). Use your phone camera at full resolution to catch more small parts.")
    g = orig.convert("L")
    g.thumbnail((1000, 1000))
    a = np.asarray(g, dtype="float32")
    if a.shape[0] > 2 and a.shape[1] > 2:
        lap = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
        if lap.var() < 40:
            tips.append("The photo looks blurry. Hold the phone steady, tap to focus and try again.")
    m = float(a.mean())
    if m < 60:
        tips.append("The photo is dark. Use bright, even light.")
    elif m > 220:
        tips.append("The photo is washed out. Avoid glare from lamps and flash.")
    return tips


# ---------------- detection ----------------
def detect_normal(path, conf):
    res = model.predict(path, conf=conf, imgsz=1024, max_det=1000, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append({"name": res.names[int(b.cls)], "conf": float(b.conf),
                    "box": (x1, y1, x2, y2)})
    return out


def detect_tiled(path, conf, tile, on_progress=None):
    sahi_model.confidence_threshold = conf
    counted_total, patched = 0, False
    if on_progress:      # count finished tiles so the page can show a real percentage
        try:
            from sahi.slicing import get_slice_bboxes
            with Image.open(path) as im:
                W, H = im.size
            n = len(get_slice_bboxes(H, W, slice_height=tile, slice_width=tile,
                                     overlap_height_ratio=0.2, overlap_width_ratio=0.2))
            counted_total = n + (1 if n > 1 else 0)   # SAHI also makes one pass over the whole photo
            real_infer, state = sahi_model.perform_inference, {"done": 0}

            def counted(*a, **k):
                out = real_infer(*a, **k)
                state["done"] += 1
                on_progress(min(state["done"], counted_total), counted_total)
                return out
            sahi_model.perform_inference = counted
            patched = True
        except Exception:
            patched = False   # progress then falls back to estimated steps on the page
    try:
        result = get_sliced_prediction(
            path, sahi_model, slice_height=tile, slice_width=tile,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
    finally:
        if patched:
            try:
                del sahi_model.perform_inference    # put the original method back
            except Exception:
                pass
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
def finalize(orig, dets, mode, job_id=None, owner=None, enhanced=None):
    purge_old()
    prev = JOBS.get(job_id, {}) if job_id else {}
    reads = prev.get("reads", [])
    if enhanced is None:
        enhanced = prev.get("enhanced", [])
    img = orig.copy()
    draw = ImageDraw.Draw(img)
    lw = max(2, max(img.size) // 500)
    rows = []
    for d in dets:
        info = INFO.get(d["name"].lower().strip(), DEFAULT)
        draw.rectangle(d["box"], outline=COLORS[level(info["hazard_fraction"])], width=lw)
        val = float(d.get("value", info["value_inr"]))  # a chip-specific price overrides the class price
        k = val / info["value_inr"] if info["value_inr"] else 1.0
        rows.append({"class": d["name"], "confidence": round(d["conf"], 2),
                     "value_inr": val, "value_low": info["value_low"] * k,
                     "value_high": info["value_high"] * k, "mass_g": info["mass_g"],
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
               "Estimated value range (INR)": f"{df.value_low.sum():.0f} to {df.value_high.sum():.0f}",
               "Estimated hazardous content": f"{hazard_pct:.1f}%",
               "Low-confidence detections": low}

    job_id = job_id or uuid.uuid4().hex[:10]
    W, H = orig.size
    pl = build_plan(summary, df, hazard_pct, mass)
    JOBS[job_id] = {"owner": owner, "ts": time.time(), "plan": pl, "reads": reads, "enhanced": enhanced, "orig": orig, "img": img, "dets": dets, "df": df,
                    "summary": summary, "metrics": metrics, "mode": mode}
    while len(JOBS) > 20:
        JOBS.popitem(last=False)
    if owner:   # keep a permanent copy so the scan can be opened again from History
        history.save(job_id, owner, orig, img, dets, mode, reads, enhanced,
                     len(df), float(df.value_inr.sum()), float(hazard_pct), pl["score"])

    shown = img.copy()
    shown.thumbnail((1600, 1600))
    buf = io.BytesIO()
    shown.save(buf, format="JPEG", quality=85)
    return {
        "job_id": job_id,
        "image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
        "classes": sorted(set(model.names.values()) | {"microcontroller"}),
        "boxes": [{"c": d["name"], "p": round(d["conf"], 3), "m": d["conf"] >= 1.0, "v": d.get("value"),
                   "val": round(rows[n]["value_inr"], 1), "mass": rows[n]["mass_g"], "hz": round(rows[n]["hazard_mass_g"], 4),
                   "b": [round(d["box"][0] / W, 5), round(d["box"][1] / H, 5),
                         round(d["box"][2] / W, 5), round(d["box"][3] / H, 5)]} for n, d in enumerate(dets)],
        "mode": mode, "low": low, "ics": sum(1 for d in dets if d["name"].lower() in ("ic", "microcontroller")),
        "metrics": {"components": len(df), "reusable": reusable,
                    "value": round(float(df.value_inr.sum())),
                    "value_low": round(float(df.value_low.sum())),
                    "value_high": round(float(df.value_high.sum())),
                    "hazard": round(float(hazard_pct), 1)},
        **pl, "quality": quality_tips(orig), "enhanced": enhanced,
        "compliance": audit(list(summary["class"])),
        "repair": guides_for(list(summary["class"])),
        "equiv": [e for e in (equiv_for(r["part"]) for r in reads if r.get("part") != "Not recognised") if e],
        "assumptions": [{"class": str(c), "value": INFO.get(str(c).lower().strip(), DEFAULT)["value_inr"],
                         "low": INFO.get(str(c).lower().strip(), DEFAULT)["value_low"],
                         "high": INFO.get(str(c).lower().strip(), DEFAULT)["value_high"],
                         "mass": INFO.get(str(c).lower().strip(), DEFAULT)["mass_g"],
                         "hazard": round(100 * INFO.get(str(c).lower().strip(), DEFAULT)["hazard_fraction"], 1),
                         "source": INFO.get(str(c).lower().strip(), DEFAULT)["source"]}
                        for c in summary["class"]],
        "summary": [{"class": str(r["class"]), "count": int(r["count"]),
                     "value": round(float(r["value_inr"]), 1), "verdict": str(r["verdict"]),
                     "level": level(INFO.get(str(r["class"]).lower().strip(), DEFAULT)["hazard_fraction"])}
                    for _, r in summary.iterrows()],
    }


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), conf: float = Form(0.2),
            tiled: bool = Form(True), tile: int = Form(640), enhance: bool = Form(True),
            progress_id: str = Form(""), user: str = Depends(current_user)):
    _prog(progress_id, user, 4, "Photo received")
    raw = file.file.read()
    img_hash = hashlib.sha256(raw).hexdigest()
    if os.environ.get("CHIPPAL_DEMO_POINTS") == "1":   # demo mode: every scan earns points, even the same photo again
        img_hash += uuid.uuid4().hex
    try:
        orig = ImageOps.exif_transpose(
            Image.open(io.BytesIO(raw))).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file is not a readable image.")
    _prog(progress_id, user, 8, "Cleaning up the photo" if enhance else "Preparing the photo")
    work, steps = enhance_if_needed(orig) if enhance else (orig, [])
    path = f"temp_{uuid.uuid4().hex[:8]}.jpg"
    work.save(path)
    _prog(progress_id, user, 12, "Getting ready to look for components")

    def tile_done(done, total):
        _prog(progress_id, user, 15 + 70 * done / total, f"Finding components (step {done} of {total})")
    try:
        with DETECT_LOCK:
            _prog(progress_id, user, 15, "Finding components")
            dets = detect_tiled(path, conf, tile, tile_done) if tiled else detect_normal(path, conf)
    finally:
        if os.path.exists(path):
            os.remove(path)
    if not dets:
        raise HTTPException(422, "No components found. Try a lower confidence.")
    _prog(progress_id, user, 88, "Estimating value and hazard")
    result = finalize(orig, dets, "tiled" if tiled else "standard", owner=user, enhanced=steps)
    _prog(progress_id, user, 97, "Almost done")
    result["salvage"] = result.get("score")   # salvage score 0-100, kept before "score" becomes the points profile
    result["score"] = award(user, img_hash)  # 10 points for each new board that finishes scanning
    return result


@app.get("/api/progress/{pid}")
def progress(pid: str, user: str = Depends(current_user)):
    p = PROGRESS.get(pid)
    if not p or p["owner"] != user:
        raise HTTPException(404, "No progress yet.")
    return {"pct": p["pct"], "stage": p["stage"]}


@app.post("/api/recompute")
def recompute(job_id: str = Form(...), dets: str = Form(...), user: str = Depends(current_user)):
    job = _job(job_id, user)
    W, H = job["orig"].size
    try:
        items = json.loads(dets)
        new = [{"name": str(i["name"]), "conf": float(i.get("conf", 1.0)),
                "box": (i["box"][0] * W, i["box"][1] * H, i["box"][2] * W, i["box"][3] * H),
                **({"value": float(i["v"])} if i.get("v") is not None else {})}
               for i in items]
    except Exception:
        raise HTTPException(400, "Bad edit data.")
    if not new:
        raise HTTPException(422, "No components left. Add one or undo your edits.")
    return finalize(job["orig"], new, job["mode"], job_id, owner=user)


@app.post("/api/chips")
def chips(job_id: str = Form(...), user: str = Depends(current_user)):
    job = _job(job_id, user)
    ics = sorted([d for d in job["dets"] if d["name"].lower().strip() in ("ic", "microcontroller")],
                 key=lambda d: -d["conf"])[:12]
    with DETECT_LOCK:
        reader = get_ocr()
        reads = [read_chip(job["orig"], d["box"], reader) for d in ics]
    out = []
    for i, (d, (text, part, score)) in enumerate(zip(ics, reads)):
        if part:
            p = PARTS[part]
            kind = str(p.get("category", "ic")).lower()
            if kind == "nan":
                kind = "ic"
            d["value"] = float(p["value_inr"])          # use the chip's own price
            d["name"] = "microcontroller" if kind == "microcontroller" else "ic"
            out.append({"n": i + 1, "text": text, "part": p["part_number"], "kind": kind,
                        "desc": p["description"], "value": float(p["value_inr"]), "match": score})
        else:
            out.append({"n": i + 1, "text": text or "(unreadable)", "part": "Not recognised",
                        "kind": str(d["name"]).lower(), "desc": "Generic value used",
                        "value": float(INFO.get(d["name"].lower(), DEFAULT)["value_inr"]), "match": 0})
    job["reads"] = out
    analysis = finalize(job["orig"], job["dets"], job["mode"], job_id, owner=user)
    return {"results": out, "analysis": analysis}


def restore_job(job_id, user):
    """Rebuild a saved scan from disk into memory. Returns the full analysis or None."""
    rec = history.load(job_id, user)
    if not rec:
        return None
    orig = Image.open(rec["orig_path"]).convert("RGB")
    dets = [{**d, "box": tuple(d["box"])} for d in rec["dets"]]
    JOBS[job_id] = {"owner": user, "ts": time.time(), "reads": rec["reads"],
                    "enhanced": rec["enhanced"]}
    return finalize(orig, dets, rec["mode"], job_id, owner=user, enhanced=rec["enhanced"])


def _job(job_id, user):
    purge_old()
    job = JOBS.get(job_id)
    if not job:
        restore_job(job_id, user)      # saved earlier? load it back from disk
        job = JOBS.get(job_id)
    if not job or job.get("owner") != user:
        raise HTTPException(404, "This scan was not found. Please analyze the photo again.")
    return job


@app.get("/api/history")
def history_list(user: str = Depends(current_user)):
    return {"scans": history.list_for(user)}


@app.get("/api/scan-image/{job_id}")
def scan_image(job_id: str, user: str = Depends(current_user)):
    """The saved original photo of a scan (used as the 'before' picture when a scan is reopened)."""
    rec = history.load(job_id, user)
    if not rec:
        raise HTTPException(404, "This scan was not found.")
    return FileResponse(rec["orig_path"], media_type="image/jpeg")


@app.post("/api/open/{job_id}")
def open_scan(job_id: str, user: str = Depends(current_user)):
    """Return the full results of a saved scan (same shape as /api/analyze). No points are given."""
    try:
        analysis = restore_job(job_id, user)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Could not rebuild this scan: {e}")
    if not analysis:
        raise HTTPException(404, "This scan was not found. It may have been deleted.")
    analysis["salvage"] = analysis.get("score")
    analysis["score"] = score_profile(user)
    return analysis


@app.get("/api/report/{job_id}.pdf")
def report_pdf(job_id: str, user: str = Depends(current_user)):
    j = _job(job_id, user)
    try:
        pdf = make_pdf(j["img"], j["summary"], j["metrics"], j["mode"],
                       plan=j["plan"]["plan"], score=j["plan"]["score"])
    except Exception as e:  # show the real reason in the terminal and in the page
        traceback.print_exc()
        raise HTTPException(500, f"Could not build the PDF: {e}")
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="chippal_report.pdf"'})


@app.get("/api/csv/{job_id}.csv")
def report_csv(job_id: str, user: str = Depends(current_user)):
    j = _job(job_id, user)
    return Response(j["df"].to_csv(index=False), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="ewaste_report.csv"'})


@app.post("/api/delete/{job_id}")
def delete_scan(job_id: str, user: str = Depends(current_user)):
    """Remove a scan from memory and from the saved history (photo and data)."""
    job = JOBS.get(job_id)
    in_memory = bool(job) and job.get("owner") == user
    if in_memory:
        JOBS.pop(job_id, None)
    if not history.delete(job_id, user) and not in_memory:
        raise HTTPException(404, "This scan was not found.")
    return {"ok": True}


@app.post("/api/delete-account")
def delete_account(response: Response, password: str = Form(""), user: str = Depends(current_user)):
    if not is_guest(user):
        if not verify_password(user, password):
            raise HTTPException(403, "That password is not correct.")
        delete_user(user)
    delete_scores(user)
    history.delete_all(user)
    for k in [k for k, v in JOBS.items() if v.get("owner") == user]:
        JOBS.pop(k, None)
    response.delete_cookie(COOKIE)
    return {"ok": True}


# ---------------- pages ----------------
@app.get("/", include_in_schema=False)
def home(request: Request):
    if not optional_user(request):
        return RedirectResponse("/login")
    return FileResponse("static/index.html")


@app.get("/login", include_in_schema=False)
def login_page(request: Request):
    if optional_user(request):
        return RedirectResponse("/")
    return FileResponse("static/login.html")