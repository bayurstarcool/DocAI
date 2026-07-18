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
        "exp": int(time.time()) + (24 * 60 * 60),
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
    return validate_token(token)

PUBLIC_PATHS = [
    "/login",
    "/api/auth/login",
    "/static",
    "/favicon.ico",
    "/data/",
]

def is_public_path(path: str) -> bool:
    return any(path.startswith(p) for p in PUBLIC_PATHS)
