"""Simple local user accounts: sign up, log in, log out.

Passwords are never stored as plain text: each one is hashed with PBKDF2-SHA256 and a random salt
(data/users.json). Each user's meals, workouts and profile live in their own file: data/users/<username>.json.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USERS = ROOT / "data" / "users.json"
USER_DIR = ROOT / "data" / "users"
LEGACY = ROOT / "data" / "user_data.json"
DEMO_USER, DEMO_PASS = "demo", "demo123"


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()


def _load() -> dict:
    if USERS.exists():
        try:
            return json.load(open(USERS))
        except Exception:  # noqa
            pass
    return {}


def _save(users: dict):
    USERS.parent.mkdir(parents=True, exist_ok=True)
    json.dump(users, open(USERS, "w"), indent=1)


def user_file(username: str) -> Path:
    return USER_DIR / f"{username}.json"


def ensure_demo():
    """First run: create the demo account and move any existing single-user data into it."""
    users = _load()
    if DEMO_USER in users:
        return
    salt = os.urandom(16)
    users[DEMO_USER] = dict(salt=salt.hex(), hash=_hash(DEMO_PASS, salt), name="Demo user")
    _save(users)
    USER_DIR.mkdir(parents=True, exist_ok=True)
    if LEGACY.exists() and not user_file(DEMO_USER).exists():
        shutil.copy(LEGACY, user_file(DEMO_USER))


def sign_up(username: str, password: str, name: str = "") -> str | None:
    """Returns an error message, or None on success."""
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{3,20}", username):
        return "Username must be 3–20 characters: letters, numbers or _."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    users = _load()
    if username in users:
        return "That username is already taken."
    salt = os.urandom(16)
    users[username] = dict(salt=salt.hex(), hash=_hash(password, salt), name=name.strip() or username)
    _save(users)
    return None


def log_in(username: str, password: str) -> bool:
    u = _load().get(username.strip().lower())
    if not u:
        return False
    return hmac.compare_digest(u["hash"], _hash(password, bytes.fromhex(u["salt"])))


def display_name(username: str) -> str:
    return _load().get(username, {}).get("name", username)
