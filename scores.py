"""Points, levels, badges and a leaderboard for ChipPal.
Every NEW board a user finishes scanning earns 10 points. The same photo only counts once,
so re-scanning one image cannot be used to farm points. Scores live on the server."""
import sqlite3
import time
from datetime import date, timedelta

from fastapi import APIRouter, Depends

from auth import DB_FILE, current_user, is_guest

POINTS_PER_SCAN = 10
LEVELS = [(0, "Newbie Sorter"), (50, "Bench Helper"), (150, "Chip Hunter"),
          (300, "Salvage Pro"), (600, "Recycling Ace"), (1000, "Eco Hero")]
BADGES = [("first", "First scan", "Finish your first scan", lambda p, s, st: s >= 1),
          ("ten", "10 scans", "Finish 10 scans", lambda p, s, st: s >= 10),
          ("fifty", "50 scans", "Finish 50 scans", lambda p, s, st: s >= 50),
          ("streak3", "3-day streak", "Scan on 3 days in a row", lambda p, s, st: st >= 3),
          ("streak7", "7-day streak", "Scan on 7 days in a row", lambda p, s, st: st >= 7),
          ("pts500", "500 points", "Reach 500 points", lambda p, s, st: p >= 500)]
router = APIRouter()


def _con():
    con = sqlite3.connect(DB_FILE)
    con.execute("CREATE TABLE IF NOT EXISTS scores (name TEXT PRIMARY KEY COLLATE NOCASE, "
                "points INTEGER NOT NULL DEFAULT 0, scans INTEGER NOT NULL DEFAULT 0, "
                "streak INTEGER NOT NULL DEFAULT 0, best_streak INTEGER NOT NULL DEFAULT 0, last_day TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS scored (name TEXT COLLATE NOCASE, hash TEXT, ts REAL, "
                "PRIMARY KEY (name, hash))")
    return con


def level_info(points):
    idx = max(i for i, (floor, _) in enumerate(LEVELS) if points >= floor)
    floor, name = LEVELS[idx]
    nxt = LEVELS[idx + 1][0] if idx + 1 < len(LEVELS) else None
    pct = 100 if nxt is None else round(100 * (points - floor) / (nxt - floor))
    return {"index": idx, "name": name, "floor": floor, "next_at": nxt,
            "to_next": None if nxt is None else nxt - points, "pct": pct}


def _badges(p, s, st):
    return [{"id": i, "name": n, "desc": d, "unlocked": bool(fn(p, s, st))} for i, n, d, fn in BADGES]


def _row(con, name):
    return con.execute("SELECT points, scans, streak, best_streak, last_day FROM scores WHERE name = ?",
                       (name,)).fetchone() or (0, 0, 0, 0, None)


def _profile(con, name, row):
    p, s, st, best, _ = row
    rank = None
    if not is_guest(name) and p > 0:
        rank = 1 + con.execute("SELECT COUNT(*) FROM scores WHERE points > ? AND name NOT LIKE 'guest-%'",
                               (p,)).fetchone()[0]
    return {"name": name, "points": p, "scans": s, "streak": st, "best_streak": best,
            "level": level_info(p), "badges": _badges(p, s, st), "rank": rank,
            "per_scan": POINTS_PER_SCAN}


def profile(name):
    con = _con()
    return _profile(con, name, _row(con, name))


def award(name, img_hash, today=None):
    """Give POINTS_PER_SCAN for a new photo. Returns the profile plus what just happened."""
    today = today or date.today()
    con = _con()
    with con:
        fresh = con.execute("INSERT OR IGNORE INTO scored (name, hash, ts) VALUES (?, ?, ?)",
                            (name, img_hash, time.time())).rowcount == 1
        row = _row(con, name)
        out = {"awarded": False, "points_added": 0, "level_up": False, "new_badges": [], "reason": None}
        if not fresh:
            out["reason"] = "You already scored this photo. Scan a new board to earn points."
            return {**_profile(con, name, row), **out}
        p, s, st, best, last = row
        before_level = level_info(p)["index"]
        before = {b["id"] for b in _badges(p, s, st) if b["unlocked"]}
        if last == today.isoformat():
            pass
        elif last == (today - timedelta(days=1)).isoformat():
            st += 1
        else:
            st = 1
        best, p, s = max(best, st), p + POINTS_PER_SCAN, s + 1
        con.execute("INSERT OR REPLACE INTO scores (name, points, scans, streak, best_streak, last_day) "
                    "VALUES (?, ?, ?, ?, ?, ?)", (name, p, s, st, best, today.isoformat()))
        out.update(awarded=True, points_added=POINTS_PER_SCAN,
                   level_up=level_info(p)["index"] > before_level,
                   new_badges=[b["name"] for b in _badges(p, s, st) if b["unlocked"] and b["id"] not in before])
        return {**_profile(con, name, (p, s, st, best, today.isoformat())), **out}


def leaderboard(limit=10):
    con = _con()
    rows = con.execute("SELECT name, points, scans FROM scores WHERE name NOT LIKE 'guest-%' AND points > 0 "
                       "ORDER BY points DESC, scans DESC, name LIMIT ?", (limit,)).fetchall()
    return [{"rank": i + 1, "name": n, "points": p, "scans": s, "level": level_info(p)["name"]}
            for i, (n, p, s) in enumerate(rows)]


def delete(name):
    with _con() as con:
        con.execute("DELETE FROM scores WHERE name = ?", (name,))
        con.execute("DELETE FROM scored WHERE name = ?", (name,))


@router.get("/api/score")
def my_score(user: str = Depends(current_user)):
    return profile(user)


@router.get("/api/leaderboard")
def board(user: str = Depends(current_user)):
    return {"top": leaderboard(10), "me": profile(user)}