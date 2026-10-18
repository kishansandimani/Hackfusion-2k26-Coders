import io
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from rapidfuzz import process, fuzz
from report import make_pdf

WEIGHTS = "weights/best.pt"


def norm(s):
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


@st.cache_resource
def load_model():
    return YOLO(WEIGHTS)


@st.cache_resource
def load_sahi_model():
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=WEIGHTS,
        confidence_threshold=0.2, device="cpu")


@st.cache_resource
def load_ocr():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


@st.cache_data
def load_db():
    db = pd.read_csv("components.csv")
    db["key"] = db["class"].str.lower().str.strip()
    return db.set_index("key").to_dict("index")


@st.cache_data
def load_parts():
    p = pd.read_csv("parts_db.csv")
    p["key"] = p["part_number"].apply(norm)
    return p.set_index("key").to_dict("index")


model = load_model()
INFO = load_db()
PARTS = load_parts()
DEFAULT = {"value_inr": 1.0, "mass_g": 0.3, "hazard_fraction": 0.03,
           "verdict": "Check manually"}


def box_color(hazard):
    if hazard >= 0.10:
        return "red"
    if hazard >= 0.04:
        return "orange"
    return "lime"


def detect_normal(path, conf):
    res = model.predict(path, conf=conf, imgsz=1024,
                        max_det=1000, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append({"name": res.names[int(b.cls)], "conf": float(b.conf),
                    "box": (x1, y1, x2, y2)})
    return out


def detect_tiled(path, conf, tile):
    sm = load_sahi_model()
    sm.confidence_threshold = conf
    result = get_sliced_prediction(
        path, sm, slice_height=tile, slice_width=tile,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
    out = []
    for p in result.object_prediction_list:
        bb = p.bbox
        out.append({"name": p.category.name, "conf": float(p.score.value),
                    "box": (bb.minx, bb.miny, bb.maxx, bb.maxy)})
    return out


@st.cache_data(show_spinner=False)
def run_detection(file_bytes, conf, use_tiles, tile):
    path = "temp.jpg"
    Image.open(io.BytesIO(file_bytes)).convert("RGB").save(path)
    return detect_tiled(path, conf, tile) if use_tiles else detect_normal(path, conf)


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
    pad = 4
    crop = orig.crop((max(0, x1 - pad), max(0, y1 - pad), x2 + pad, y2 + pad))
    crop = crop.convert("L")
    w, h = crop.size
    scale = max(1, 300 // max(1, min(w, h)))
    if scale > 1:
        crop = crop.resize((w * scale, h * scale), Image.LANCZOS)
    best = ("", None, 0)
    for angle in (0, 90, 180, 270):
        arr = np.array(crop.rotate(angle, expand=True))
        txts = reader.readtext(
            arr, detail=0,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
        text = " ".join(txts)
        part, score = match_part(text)
        if (score, len(text)) > (best[2], len(best[0])):
            best = (text, part, score)
    return best


st.set_page_config(page_title="E-Waste Circuit Analyzer", layout="wide")
st.title("E-Waste Circuit Component Analyzer")
st.caption("Box colours show hazard level: red = high, orange = medium, green = low.")

col_a, col_b, col_c = st.columns(3)
conf = col_a.slider("Confidence threshold", 0.05, 0.9, 0.2)
use_tiles = col_b.checkbox("Tiled detection (finds more small parts, slower)", value=True)
tile = col_c.select_slider("Tile size (px)", options=[512, 640, 800], value=640)

file = st.file_uploader("Upload a circuit board image", type=["jpg", "jpeg", "png"])

if file:
    file_bytes = file.getvalue()
    orig = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    with st.spinner("Detecting components..."):
        dets = run_detection(file_bytes, conf, use_tiles, tile)

    img = orig.copy()
    draw = ImageDraw.Draw(img)
    rows = []
    for d in dets:
        info = INFO.get(d["name"].lower().strip(), DEFAULT)
        draw.rectangle(d["box"], outline=box_color(info["hazard_fraction"]), width=2)
        rows.append({
            "class": d["name"],
            "confidence": round(d["conf"], 2),
            "value_inr": info["value_inr"],
            "mass_g": info["mass_g"],
            "hazard_mass_g": info["mass_g"] * info["hazard_fraction"],
            "verdict": info["verdict"],
        })

    left, right = st.columns([3, 2])

    with left:
        mode = "tiled" if use_tiles else "standard"
        st.image(img, caption=f"{len(rows)} components detected ({mode} mode)",
                 use_container_width=True)

    with right:
        if rows:
            df = pd.DataFrame(rows)
            total_mass = df.mass_g.sum()
            hazard_pct = 100 * df.hazard_mass_g.sum() / total_mass if total_mass else 0
            reusable = (df.verdict.str.contains("Reuse")).sum()

            c1, c2 = st.columns(2)
            c1.metric("Components", len(df))
            c2.metric("Reusable parts", int(reusable))
            c3, c4 = st.columns(2)
            c3.metric("Est. value (INR)", f"{df.value_inr.sum():.0f}")
            c4.metric("Est. hazardous %", f"{hazard_pct:.1f}%")

            summary = (df.groupby("class")
                         .agg(count=("class", "size"),
                              value_inr=("value_inr", "sum"),
                              verdict=("verdict", "first"))
                         .reset_index()
                         .sort_values("count", ascending=False))
            st.subheader("Component inventory")
            st.dataframe(summary, use_container_width=True, hide_index=True)

            low = (df.confidence < 0.4).sum()
            if low:
                st.info(f"{low} detections have low confidence (below 0.40) "
                        "and should be checked by a person.")

            st.download_button("Download CSV report", df.to_csv(index=False),
                               file_name="ewaste_report.csv", mime="text/csv")
            
            pdf_bytes = make_pdf(
                img, summary,
                {"Components detected": len(df),
                 "Reusable parts": int(reusable),
                 "Estimated value (INR)": f"{df.value_inr.sum():.0f}",
                 "Estimated hazardous content": f"{hazard_pct:.1f}%",
                 "Low-confidence detections": int(low)},
                "tiled" if use_tiles else "standard")
            
            st.download_button("Download PDF report", pdf_bytes,
                               file_name="ewaste_report.pdf", mime="application/pdf")
            st.caption("Value and hazard figures are estimates from a lookup "
                       "table, not measurements.")
        else:
            st.warning("No components found. Try lowering the confidence slider.")

    # ---------------- Chip identification ----------------
    st.divider()
    st.subheader("Chip identification (OCR)")
    st.caption("Reads the printed marking on detected ICs and matches it to a "
               "parts database. Works best on sharp, close-up photos.")

    ic_dets = sorted([d for d in dets if d["name"].lower().strip() == "ic"],
                     key=lambda d: -d["conf"])[:12]

    if not ic_dets:
        st.write("No ICs detected in this image.")
    elif st.button(f"Read chip markings ({len(ic_dets)} ICs)"):
        reader = load_ocr()
        results = []
        prog = st.progress(0.0)
        for i, d in enumerate(ic_dets):
            text, part, score = read_chip(orig, d["box"], reader)
            if part:
                info = PARTS[part]
                results.append({"IC #": i + 1, "text read": text,
                                "identified as": info["part_number"],
                                "description": info["description"],
                                "value_inr": info["value_inr"],
                                "match %": score})
            else:
                results.append({"IC #": i + 1, "text read": text or "(unreadable)",
                                "identified as": "Not recognised",
                                "description": "Generic IC value used",
                                "value_inr": INFO.get("ic", DEFAULT)["value_inr"],
                                "match %": 0})
            prog.progress((i + 1) / len(ic_dets))
        res_df = pd.DataFrame(results)
        st.dataframe(res_df, use_container_width=True, hide_index=True)
        found = (res_df["identified as"] != "Not recognised").sum()
        st.success(f"Identified {found} of {len(res_df)} ICs from their markings.")