"""Authentication, roles, rate limiting, headers, storage safety, versions."""

from __future__ import annotations

import time

import jwt
import pytest
from sqlalchemy import select

from ngo_agent.errors import Forbidden
from ngo_agent.models import User
from ngo_agent.security import AuthError, RateLimiter, decode_token, hash_password, issue_token, jwt_secret, verify_password
from ngo_agent.storage import LocalStorage, StorageError
from ngo_agent.versions import config_digest
from conftest import PW, SECRET


def test_passwords_are_hashed_with_a_salt_and_verified():
    h1, h2 = hash_password(PW), hash_password(PW)
    assert h1 != h2 and h1.startswith("scrypt$") and PW not in h1
    assert verify_password(PW, h1) and not verify_password("wrong password!", h1) and not verify_password(PW, "garbage") and not verify_password(PW, "scrypt$1$2$3$4$5")
    with pytest.raises(ValueError):
        hash_password("short")


def test_tokens_expire_and_are_signed():
    t = issue_token("u", "o", "REVIEWER", SECRET, lifetime_s=60)
    assert decode_token(t, SECRET)["org"] == "o"
    with pytest.raises(AuthError):
        decode_token(t, "y" * 40)
    with pytest.raises(AuthError):
        decode_token(issue_token("u", "o", "REVIEWER", SECRET, lifetime_s=-5), SECRET)
    forged = jwt.encode({"sub": "u", "org": "o", "exp": int(time.time()) + 60}, "", algorithm="none")
    with pytest.raises(AuthError):
        decode_token(forged, SECRET)
    with pytest.raises(AuthError):
        jwt_secret("too short")


def test_rate_limiter_blocks_after_the_limit_and_recovers():
    now = [0.0]
    rl = RateLimiter(limit=3, window_s=10, clock=lambda: now[0])
    for _ in range(3):
        assert not rl.blocked("a")
        rl.record_failure("a")
    assert rl.blocked("a") and not rl.blocked("b")
    now[0] = 11.0
    assert not rl.blocked("a")


def test_api_requires_a_token_and_login_is_generic_and_rate_limited(P):
    for url in ("/photos", "/me", "/photos/x", "/evaluations/x", "/organisations/x/feedback"):
        assert P.client.get(url).status_code == 401
    assert P.client.get("/photos", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    bad_user = P.client.post("/auth/login", json={"email": "nobody@example.org", "password": "whatever long"})
    bad_pw = P.client.post("/auth/login", json={"email": "rev_a@example.org", "password": "whatever long"})
    assert bad_user.status_code == bad_pw.status_code == 401 and bad_user.json() == bad_pw.json()   # the same answer: no account enumeration
    for _ in range(5):
        P.client.post("/auth/login", json={"email": "rev_a@example.org", "password": "still wrong x"})
    assert P.client.post("/auth/login", json={"email": "rev_a@example.org", "password": PW}).status_code == 429
    assert P.client.get("/health").json() == {"status": "ok"}


def test_deactivating_a_user_takes_effect_immediately(P):
    h = P.login("rev_a")
    assert P.client.get("/me", headers=h).status_code == 200
    with P.factory() as s:
        s.scalar(select(User).where(User.email == "rev_a@example.org")).is_active = False
        s.commit()
    assert P.client.get("/me", headers=h).status_code == 401
    assert P.client.post("/auth/login", json={"email": "rev_a@example.org", "password": PW}).status_code == 401


def test_roles_limit_what_each_person_can_do(P):
    hr, ha = P.login("rev_a"), P.login("admin_a")
    assert P.client.post(f"/organisations/{P.orgA_id}/users", json={"email": "z@example.org", "password": "a long password", "role": "REVIEWER"}, headers=hr).status_code == 403
    assert P.client.get(f"/organisations/{P.orgA_id}/learning-examples", headers=hr).status_code == 403
    assert P.client.get(f"/organisations/{P.orgA_id}/learning-examples", headers=ha).status_code == 200
    assert P.client.post(f"/organisations/{P.orgA_id}/versions", json={"kind": "prompt", "version_number": "x"}, headers=hr).status_code == 403
    assert P.client.post(f"/organisations/{P.orgA_id}/users", json={"email": "z@example.org", "password": "short", "role": "REVIEWER"}, headers=ha).status_code == 422
    assert P.client.post(f"/organisations/{P.orgA_id}/users", json={"email": "z@example.org", "password": "a long password", "role": "KING"}, headers=ha).status_code == 422
    assert P.client.post(f"/organisations/{P.orgA_id}/users", json={"email": "rev_a@example.org", "password": "a long password", "role": "REVIEWER"}, headers=ha).status_code == 409
    ok = P.client.post(f"/organisations/{P.orgA_id}/users", json={"email": "Z@Example.org", "password": "a long password", "role": "REVIEWER"}, headers=ha)
    assert ok.status_code == 201 and ok.json()["email"] == "z@example.org"
    assert "password" not in ok.text
    assert P.client.post(f"/organisations/{P.orgB_id}/users", json={"email": "q@example.org", "password": "a long password", "role": "REVIEWER"}, headers=ha).status_code == 404


def test_security_headers_and_no_caching(P):
    r = P.client.get("/me", headers=P.login("rev_a"))
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store" and r.headers["x-frame-options"] == "DENY"
    page = P.client.get("/")
    assert page.status_code == 200 and "default-src 'self'" in page.headers["content-security-policy"] and "innerHTML" not in P.client.get("/static/app.js").text
    # scripts and styles must be revalidated on every load, or an upgraded server would feed a browser's cached old script to the new API
    for path in ("/static/app.js", "/static/style.css"):
        assert P.client.get(path).headers["cache-control"] == "no-cache", path
    assert page.headers["cache-control"] == "no-store"


def test_every_script_inserts_text_only_and_the_page_has_no_inline_script_or_style(P):
    """The interface (photo review and the administrator screens) must never turn data into markup, and the content-security policy forbids inline script and style."""
    import re
    from pathlib import Path

    static = Path(__file__).resolve().parents[1] / "src" / "ngo_agent" / "api" / "static"
    scripts = sorted(static.glob("*.js"))
    assert {p.name for p in scripts} >= {"app.js", "admin.js"}
    unsafe = re.compile(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function|\.style\.|setAttribute\(\s*[\"']style|srcdoc|javascript:")
    for p in scripts:
        assert not unsafe.search(p.read_text(encoding="utf-8")), p.name
    page = P.client.get("/")
    html = page.text
    assert re.findall(r'<script[^>]*src="(/static/[^"]+)"', html) == ["/static/app.js", "/static/admin.js"] and not re.search(r"<script(?![^>]*\bsrc=)", html)
    assert "style=" not in html and "<style" not in html and not re.search(r"\son\w+=", html)                  # no inline handlers either
    assert "script-src 'self'" in page.headers["content-security-policy"] and "unsafe-inline" not in page.headers["content-security-policy"]
    for name in ("app.js", "admin.js", "style.css"):
        r = P.client.get(f"/static/{name}")
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache", name


def test_the_administrator_screens_get_their_data_only_from_administrator_endpoints(P):
    """Showing or hiding the navigation is a convenience; the real protection is the server, which every screen's data goes through."""
    hr, ha, a = P.login("rev_a"), P.login("admin_a"), P.orgA_id
    for path in ("learning-loop", "changelog", "few-shot", "gap-analyses"):
        assert P.client.get(f"/organisations/{a}/{path}", headers=hr).status_code == 403, path
        assert P.client.get(f"/organisations/{a}/{path}", headers=ha).status_code == 200, path
    assert "ORG_ADMIN" not in P.client.get("/static/admin.js").text           # admin.js holds no role logic: it only shows what the server returns to an administrator


def test_image_endpoint_needs_authentication_and_returns_the_image(P):
    pid = P.upload().json()["photo"]["photo_id"]
    assert P.client.get(f"/photos/{pid}/image").status_code == 401
    r = P.client.get(f"/photos/{pid}/image", headers=P.login("rev_a"))
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.headers["cache-control"] == "private, no-store" and r.content[:2] == b"\xff\xd8"


def test_errors_never_leak_internals(P):
    r = P.client.get("/photos/not-a-photo", headers=P.login("rev_a"))
    assert r.status_code == 404 and set(r.json()) == {"error", "detail"} and ".py" not in r.text and "Traceback" not in r.text


def test_local_storage_rejects_path_traversal_and_is_write_once(tmp_path):
    st = LocalStorage(tmp_path / "s")
    for bad in ("../x", "a/../../x", "/abs", "a b", "a\\b", ""):
        with pytest.raises(StorageError):
            st.put(bad, b"x")
    st.put("org/images/h.jpg", b"first")
    st.put("org/images/h.jpg", b"second")
    assert st.get("org/images/h.jpg") == b"first" and st.exists("org/images/h.jpg") and not st.exists("org/images/none.jpg")
    with pytest.raises(StorageError):
        st.get("org/images/none.jpg")


def test_version_digest_changes_with_the_rulebook_and_bootstrap_registered_both_versions(P):
    assert len(config_digest()) == 64
    ha = P.login("admin_a")
    v = P.client.get(f"/organisations/{P.orgA_id}/versions", headers=ha).json()
    by = {x["configuration"]["profile"]: x for x in v["rulebook_versions"]}
    assert by["layered_proposed_v1"]["approval_status"] == "APPROVED" and by["layered_proposed_v2"]["approval_status"] == "DRAFT"
    assert v["active_rulebook_version_id"] == by["layered_proposed_v1"]["version_id"] and v["active_prompt_version_id"]
    assert by["layered_proposed_v1"]["configuration"]["config_digest"] == config_digest() and by["layered_proposed_v1"]["configuration"]["bootstrap_approval"] is True
    assert {x["configuration"]["prompt_hash"] for x in v["prompt_versions"]} >= {"11427c79d96c569a"}
