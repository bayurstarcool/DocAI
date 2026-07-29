"""
DocAI Authentication System
"""

import time
import json
import base64
import os
from fastapi import Request

ADMIN_CREDENTIALS = {
    "username": os.getenv("ADMIN_USER", "admin"),
    "password": os.getenv("ADMIN_PASSWORD", "change-me"),
}

def create_token(username: str) -> str:
    payload = {
        "user": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + (30 * 24 * 60 * 60),
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()

def validate_token(token: str) -> bool:
    try:
        # Remove quotes if present
        token = token.strip('"').strip("'")
        payload = json.loads(base64.b64decode(token))
        if payload.get("user") != ADMIN_CREDENTIALS["username"]:
            return False
        if payload.get("exp", 0) < time.time():
            return False
        return True
    except Exception:
        return False

def get_token_from_request(request: Request) -> str:
    # Check Authorization header first
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip('"').strip("'")
    
    # Check cookie
    token = request.cookies.get("docai_token")
    if token:
        return token.strip('"').strip("'")
    
    return ""

def is_authenticated(request: Request) -> bool:
    token = get_token_from_request(request)
    if not token:
        return False
    if validate_token(token):
        return True
    return bool(validate_token_any(token))

PUBLIC_PATHS = [
    "/login",
    "/api/auth/login",
    "/static",
    "/favicon.ico",
    "/data/",
]

def is_public_path(path: str) -> bool:
    return any(path.startswith(p) for p in PUBLIC_PATHS)

# Simple user DB for interns (JSON file)
import hashlib

INTERN_DB_PATH = os.path.join(os.path.dirname(__file__), "interns.json")

def _load_interns():
    if not os.path.exists(INTERN_DB_PATH):
        return {}
    with open(INTERN_DB_PATH) as f:
        return json.load(f)

def _save_interns(data):
    with open(INTERN_DB_PATH, "w") as f:
        json.dump(data, f)

def register_intern(username: str, password: str) -> dict:
    if not username or not password:
        return {"error": "Username dan password wajib diisi"}
    if len(password) < 4:
        return {"error": "Password minimal 4 karakter"}
    interns = _load_interns()
    if username in interns:
        return {"error": "Username sudah terdaftar"}
    h = hashlib.sha256(password.encode()).hexdigest()
    interns[username] = {"password": h, "created_at": time.time()}
    _save_interns(interns)
    return {"success": True, "username": username}

def validate_intern(username: str, password: str) -> bool:
    interns = _load_interns()
    if username not in interns:
        return False
    h = hashlib.sha256(password.encode()).hexdigest()
    return interns[username]["password"] == h

VALIDATE_USERS = [ADMIN_CREDENTIALS["username"]]

def validate_token_any(token: str) -> str:
    """Return username if token valid for any user, else empty string."""
    try:
        token = token.strip(chr(34)).strip(chr(39))
        payload = json.loads(base64.b64decode(token))
        user = payload.get("user", "")
        exp = payload.get("exp", 0)
        if exp < time.time():
            return ""
        # Check admin
        if user == ADMIN_CREDENTIALS["username"]:
            return user
        # Check intern
        interns = _load_interns()
        if user in interns:
            return user
        return ""
    except Exception:
        return ""


# Simple user DB for interns (JSON file)
import hashlib

INTERN_DB_PATH = os.path.join(os.path.dirname(__file__), "interns.json")

def _load_interns():
    if not os.path.exists(INTERN_DB_PATH):
        return {}
    with open(INTERN_DB_PATH) as f:
        return json.load(f)

def _save_interns(data):
    with open(INTERN_DB_PATH, "w") as f:
        json.dump(data, f)

def register_intern(username: str, password: str) -> dict:
    if not username or not password:
        return {"error": "Username dan password wajib diisi"}
    if len(password) < 4:
        return {"error": "Password minimal 4 karakter"}
    interns = _load_interns()
    if username in interns:
        return {"error": "Username sudah terdaftar"}
    h = hashlib.sha256(password.encode()).hexdigest()
    interns[username] = {"password": h, "created_at": time.time()}
    _save_interns(interns)
    return {"success": True, "username": username}

def validate_intern(username: str, password: str) -> bool:
    interns = _load_interns()
    if username not in interns:
        return False
    h = hashlib.sha256(password.encode()).hexdigest()
    return interns[username]["password"] == h

def validate_token_any(token: str) -> str:
    """Return username if token valid for any user, else empty string."""
    try:
        token = token.strip('"').strip("'")
        payload = json.loads(base64.b64decode(token))
        user = payload.get("user", "")
        exp = payload.get("exp", 0)
        if exp < time.time():
            return ""
        # Check admin
        if user == ADMIN_CREDENTIALS["username"]:
            return user
        # Check intern
        interns = _load_interns()
        if user in interns:
            return user
        return ""
    except Exception:
        return ""
