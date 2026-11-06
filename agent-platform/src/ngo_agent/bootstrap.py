"""Create an organisation with its first administrator and the registered engine versions (V1 approved and active, V2 draft)."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from layered_engine.rulebook import Rulebook

from .enums import Role
from .errors import Conflict
from .models import Organisation, User
from .security import hash_password
from .versions import register_engine_versions


def create_organisation(s: Session, name: str, admin_email: str, admin_password: str, admin_name: str = "", configuration: Optional[dict] = None,
                        rb: Optional[Rulebook] = None, register_versions: bool = True) -> tuple[Organisation, User]:
    if s.scalar(select(Organisation).where(Organisation.name == name)):
        raise Conflict(f"organisation {name!r} already exists")
    email = admin_email.strip().lower()
    if s.scalar(select(User).where(User.email == email)):
        raise Conflict("that email is already registered")
    org = Organisation(name=name, configuration=configuration or {})
    s.add(org)
    s.flush()
    admin = User(organisation_id=org.organisation_id, email=email, display_name=admin_name or email, password_hash=hash_password(admin_password), role=Role.ORG_ADMIN.value)
    s.add(admin)
    s.flush()
    if register_versions:
        register_engine_versions(s, org.organisation_id, admin.user_id, rb)
    return org, admin


def add_user(s: Session, org: Organisation, email: str, password: str, role: str, display_name: str = "") -> User:
    u = User(organisation_id=org.organisation_id, email=email.strip().lower(), display_name=display_name or email, password_hash=hash_password(password), role=role)
    s.add(u)
    s.flush()
    return u
