"""Simple login for ChipPal: SQLite users, PBKDF2 password hashes, signed cookie sessions.
Uses only the Python standard library plus FastAPI."""
import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time

from fastapi import APIRouter, Form, HTTPException, Request, Response

DB_FILE = "users.db"
COOKIE = "chippal_session"
SESSION_SECONDS = 7 * 24 * 3600
# People need this code to create an account. Change it, or set CHIPPAL_INVITE.
INVITE_CODE = os.environ.get("CHIPPAL_INVITE", "CHIPPAL2026")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,20}$")
FAILS = {}  # (ip, username) -> [count, locked_until]
router = APIRouter()


def _secret():
    env = os.environ.get("CHIPPAL_SECRET")
    if env:
        return env.encode()
    if os.path.exists(".secret"):
        return open(".secret", "rb").read().strip()
    key = secrets.token_hex(32).encode()
    with open(".secret", "wb") as f:
        f.write(key)
    return key


SECRET = _secret()


def _db():
    con = sqlite3.connect(DB_FILE)
    con.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, "
                "name TEXT UNIQUE COLLATE NOCASE, salt TEXT, hash TEXT, created REAL)")
    return con


def hash_pw(password, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 200_000).hex()


def make_token(name, now=None):
    exp = int((now or time.time()) + SESSION_SECONDS)
    payload = f"{name}|{exp}"
    sig = hmac.new(SECRET, payload.encode(), "sha256").hexdigest()
    return base64.urlsafe_b64encode(f"{payload}|{sig}".encode()).decode()


def read_token(token, now=None):
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        payload, sig = raw.rsplit("|", 1)
        good = hmac.new(SECRET, payload.encode(), "sha256").hexdigest()
        name, exp = payload.rsplit("|", 1)
        if hmac.compare_digest(sig, good) and int(exp) > (now or time.time()):
            return name
    except Exception:
        pass
    return None


def optional_user(request: Request):
    tok = request.cookies.get(COOKIE)
    return read_token(tok) if tok else None


def current_user(request: Request):
    name = optional_user(request)
    if not name:
        raise HTTPException(401, "Please sign in.")
    return name


def _set_cookie(request, response, name):
    response.set_cookie(COOKIE, make_token(name), max_age=SESSION_SECONDS, httponly=True,
                        samesite="lax", secure=(request.url.scheme == "https"))


def _check_locked(key):
    rec = FAILS.get(key)
    if rec and rec[1] > time.time():
        raise HTTPException(429, "Too many attempts. Wait a minute and try again.")


def _fail(key):
    rec = FAILS.setdefault(key, [0, 0])
    rec[0] += 1
    if rec[0] >= 5:
        rec[0], rec[1] = 0, time.time() + 60


@router.post("/api/register")
def register(request: Request, response: Response, username: str = Form(...),
             password: str = Form(...), invite: str = Form(...)):
    ip = request.client.host if request.client else "?"
    _check_locked((ip, "register"))
    if not hmac.compare_digest(invite.strip(), INVITE_CODE):
        _fail((ip, "register"))
        raise HTTPException(403, "That team code is not correct.")
    if not NAME_RE.match(username):
        raise HTTPException(400, "Username: 3 to 20 letters, numbers, dot, dash or underscore.")
    if len(password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")
    salt = secrets.token_hex(16)
    try:
        with _db() as con:
            con.execute("INSERT INTO users (name, salt, hash, created) VALUES (?,?,?,?)",
                        (username, salt, hash_pw(password, salt), time.time()))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "That username is taken.")
    _set_cookie(request, response, username)
    return {"name": username}


@router.post("/api/login")
def login(request: Request, response: Response, username: str = Form(...), password: str = Form(...)):
    ip = request.client.host if request.client else "?"
    key = (ip, username.lower())
    _check_locked(key)
    row = _db().execute("SELECT name, salt, hash FROM users WHERE name = ?", (username,)).fetchone()
    salt = row[1] if row else "00" * 16
    ok = hmac.compare_digest(hash_pw(password, salt), row[2] if row else "x")
    if not (row and ok):
        _fail(key)
        raise HTTPException(401, "Wrong username or password.")
    FAILS.pop(key, None)
    _set_cookie(request, response, row[0])
    return {"name": row[0]}


@router.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE)
    return {"ok": True}


@router.get("/api/me")
def me(request: Request):
    return {"name": current_user(request)}