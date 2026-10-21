"""Saved scans for ChipPal.

Each analysis is stored on disk so it survives restarts and can be reopened later:
  data/history.db          one row per scan (numbers, detections, chip readings)
  data/scans/<id>.jpg      the original photo
  data/scans/<id>_t.jpg    a small thumbnail

Set the environment variable CHIPPAL_DATA to keep the data somewhere else
(for example a persistent disk on your hosting service).
"""
import base64
import json
import os
import re
import sqlite3
import threading
import time

DATA = os.environ.get("CHIPPAL_DATA", "data")
IMG_DIR = os.path.join(DATA, "scans")
DB_FILE = os.path.join(DATA, "history.db")
MAX_PER_USER = 30            # oldest scans are removed beyond this
ID_RE = re.compile(r"^[0-9a-f]{6,32}$")
_lock = threading.Lock()
os.makedirs(IMG_DIR, exist_ok=True)


def _con():
    con = sqlite3.connect(DB_FILE)
    con.execute("CREATE TABLE IF NOT EXISTS scans ("
                "id TEXT PRIMARY KEY, owner TEXT, created REAL, updated REAL, mode TEXT, "
                "components INTEGER, value REAL, hazard REAL, score INTEGER, "
                "dets TEXT, reads TEXT, enhanced TEXT)")
    con.execute("CREATE INDEX IF NOT EXISTS scans_owner ON scans(owner, created)")
    return con


def _path(scan_id, thumb=False):
    return os.path.join(IMG_DIR, scan_id + ("_t.jpg" if thumb else ".jpg"))


def _remove_files(scan_id):
    for p in (_path(scan_id), _path(scan_id, True)):
        if os.path.exists(p):
            os.remove(p)


def save(scan_id, owner, orig, annotated, dets, mode, reads, enhanced,
         components, value, hazard, score):
    """Create or update a scan. The created time is kept when updating."""
    if not ID_RE.match(scan_id or ""):
        return
    with _lock:
        if not os.path.exists(_path(scan_id)):
            orig.convert("RGB").save(_path(scan_id), format="JPEG", quality=92)
        thumb = annotated.convert("RGB")
        thumb.thumbnail((360, 360))
        thumb.save(_path(scan_id, True), format="JPEG", quality=80)
        now = time.time()
        with _con() as con:
            con.execute(
                "INSERT INTO scans (id, owner, created, updated, mode, components, value, hazard, score, dets, reads, enhanced) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET updated=excluded.updated, mode=excluded.mode, "
                "components=excluded.components, value=excluded.value, hazard=excluded.hazard, "
                "score=excluded.score, dets=excluded.dets, reads=excluded.reads, enhanced=excluded.enhanced",
                (scan_id, owner, now, now, mode, int(components), float(value), float(hazard), int(score),
                 json.dumps(dets), json.dumps(reads or []), json.dumps(enhanced or [])))
            old = con.execute("SELECT id FROM scans WHERE owner=? ORDER BY created DESC LIMIT -1 OFFSET ?",
                              (owner, MAX_PER_USER)).fetchall()
            for (oid,) in old:
                con.execute("DELETE FROM scans WHERE id=?", (oid,))
                _remove_files(oid)


def list_for(owner):
    with _lock, _con() as con:
        rows = con.execute("SELECT id, created, mode, components, value, hazard, score FROM scans "
                           "WHERE owner=? ORDER BY created DESC", (owner,)).fetchall()
    out = []
    for sid, created, mode, comps, value, hazard, score in rows:
        thumb = ""
        p = _path(sid, True)
        if os.path.exists(p):
            with open(p, "rb") as f:
                thumb = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
        out.append({"id": sid, "ts": created, "mode": mode, "components": comps,
                    "value": round(value), "hazard": round(hazard, 1), "score": score, "thumb": thumb})
    return out


def load(scan_id, owner):
    """Return the saved scan (only for its owner) or None."""
    if not ID_RE.match(scan_id or ""):
        return None
    with _lock, _con() as con:
        row = con.execute("SELECT mode, dets, reads, enhanced FROM scans WHERE id=? AND owner=?",
                          (scan_id, owner)).fetchone()
    if not row or not os.path.exists(_path(scan_id)):
        return None
    return {"mode": row[0], "dets": json.loads(row[1]), "reads": json.loads(row[2]),
            "enhanced": json.loads(row[3]), "orig_path": _path(scan_id)}


def delete(scan_id, owner):
    if not ID_RE.match(scan_id or ""):
        return False
    with _lock, _con() as con:
        cur = con.execute("DELETE FROM scans WHERE id=? AND owner=?", (scan_id, owner))
        if cur.rowcount:
            _remove_files(scan_id)
        return bool(cur.rowcount)


def delete_all(owner):
    with _lock, _con() as con:
        ids = [r[0] for r in con.execute("SELECT id FROM scans WHERE owner=?", (owner,)).fetchall()]
        con.execute("DELETE FROM scans WHERE owner=?", (owner,))
        for sid in ids:
            _remove_files(sid)