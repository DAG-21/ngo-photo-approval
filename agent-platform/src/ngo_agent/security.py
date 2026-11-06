"""Passwords (scrypt), tokens (JWT HS256) and a small login rate limiter. No secret is ever logged or defaulted in production."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict, deque
from typing import Any, Optional

import jwt

TOKEN_LIFETIME_S = 8 * 3600
_N, _R, _P = 2**14, 8, 1
MIN_PASSWORD_LENGTH = 10


class AuthError(Exception):
    pass


def hash_password(password: str) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"password must have at least {MIN_PASSWORD_LENGTH} characters")
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(_N, _R, _P, base64.b64encode(salt).decode(), base64.b64encode(dk).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, dk = stored.split("$")
        if scheme != "scrypt":
            return False
        calc = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc, base64.b64decode(dk))
    except Exception:  # noqa: BLE001  (malformed hash: treat as a failed login, never as an error that reveals why)
        return False


def jwt_secret(explicit: Optional[str] = None) -> str:
    s = explicit or os.environ.get("AGENT_JWT_SECRET")
    if not s or len(s) < 32:
        raise AuthError("AGENT_JWT_SECRET must be set to at least 32 characters")
    return s


def issue_token(user_id: str, organisation_id: str, role: str, secret: str, lifetime_s: int = TOKEN_LIFETIME_S) -> str:
    t = int(time.time())
    return jwt.encode({"sub": user_id, "org": organisation_id, "role": role, "iat": t, "exp": t + lifetime_s}, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["exp", "sub", "org"]})
    except jwt.PyJWTError as exc:
        raise AuthError("invalid or expired token") from exc


class RateLimiter:
    """At most `limit` failed attempts per key within `window_s` seconds. In memory: one instance only (see the design note)."""

    def __init__(self, limit: int = 5, window_s: int = 300, clock=time.monotonic) -> None:  # noqa: ANN001
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._fails: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, key: str) -> deque[float]:
        q = self._fails[key]
        while q and self.clock() - q[0] > self.window_s:
            q.popleft()
        return q

    def blocked(self, key: str) -> bool:
        return len(self._prune(key)) >= self.limit

    def record_failure(self, key: str) -> None:
        self._prune(key).append(self.clock())

    def reset(self, key: str) -> None:
        self._fails.pop(key, None)
