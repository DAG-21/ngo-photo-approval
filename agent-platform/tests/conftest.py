from __future__ import annotations

import io
import os
import random
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.engine import make_url

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "layered-engine" / "src"))

from ngo_agent.api.app import create_app  # noqa: E402
from ngo_agent.bootstrap import add_user, create_organisation  # noqa: E402
from ngo_agent.db import create_schema, make_engine, make_session_factory  # noqa: E402
from ngo_agent.evaluators import ScriptedEvaluator  # noqa: E402
from ngo_agent.storage import LocalStorage  # noqa: E402
from layered_engine.rulebook import Rulebook  # noqa: E402

SECRET = "x" * 40
PW = "correct horse battery"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """No test may ever create a real Gemini client."""
    try:
        from google import genai
    except ImportError:
        return

    def boom(*_a, **_k):
        raise AssertionError("tests must never create a real Gemini client")

    monkeypatch.setattr(genai, "Client", boom)


@pytest.fixture(scope="session")
def rb() -> Rulebook:
    return Rulebook()


def jpeg_bytes(seed: int = 0, size=(160, 120)) -> bytes:
    rnd = random.Random(seed)
    im = Image.new("RGB", size, (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    for _ in range(40):
        x, y = rnd.randrange(size[0]), rnd.randrange(size[1])
        im.putpixel((x, y), (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    return buf.getvalue()


class Platform:
    """One test platform: a file-backed SQLite database, local storage, a scripted evaluator, and two organisations."""

    def __init__(self, tmp: Path, rb: Rulebook, embedder=None, gap_analyzer=None, note_drafter=None, rule_drafter=None, reason_summarizer=None) -> None:  # noqa: ANN001
        self.embedder = embedder
        self.gap_analyzer = gap_analyzer
        self._admin_engine = None
        self._pg_schema = None
        pg_url = os.environ.get("TEST_DATABASE_URL")
        if pg_url:
            # PostgreSQL: every test gets its own schema (tables, triggers and the trigger function all live in it)
            self._pg_schema = "t_" + uuid.uuid4().hex[:12]
            self._admin_engine = make_engine(pg_url)
            with self._admin_engine.begin() as c:
                c.exec_driver_sql(f"CREATE SCHEMA {self._pg_schema}")
            url = make_url(pg_url).update_query_dict({"options": f"-csearch_path={self._pg_schema}"})
            self.engine = make_engine(url.render_as_string(hide_password=False))
        else:
            self.engine = make_engine(f"sqlite:///{(tmp / 'test.db').as_posix()}")
        if os.environ.get("TEST_USE_ALEMBIC"):
            # build the schema with the production migration instead of create_all
            from alembic import command
            from alembic.config import Config

            root = Path(__file__).resolve().parents[1]
            cfg = Config(str(root / "alembic.ini"))
            cfg.set_main_option("script_location", str(root / "alembic"))
            saved = os.environ.get("DATABASE_URL")
            os.environ["DATABASE_URL"] = self.engine.url.render_as_string(hide_password=False)
            try:
                command.upgrade(cfg, "head")
            finally:
                if saved is None:
                    os.environ.pop("DATABASE_URL", None)
                else:
                    os.environ["DATABASE_URL"] = saved
        else:
            create_schema(self.engine)
        self.factory = make_session_factory(self.engine)
        self.storage = LocalStorage(tmp / "storage")
        self.evaluator = ScriptedEvaluator(rulebook=rb)
        self.rb = rb
        self.app = create_app(self.factory, self.storage, self, SECRET, run_inline=True, activity_ids=set(rb.resolve("layered_proposed_v1").activity_ids), embedder=embedder,
                              gap_analyzer=gap_analyzer, rulebook=rb, note_drafter=note_drafter, rule_drafter=rule_drafter, reason_summarizer=reason_summarizer)
        self.client = TestClient(self.app, raise_server_exceptions=False)
        self.script = {}   # per test: kwargs for the next ScriptedEvaluator
        with self.factory() as s:
            self.orgA, self.adminA = create_organisation(s, "Org A", "admin-a@example.org", PW, rb=rb)
            self.orgB, self.adminB = create_organisation(s, "Org B", "admin-b@example.org", PW, rb=rb)
            self.users = {"admin_a": self.adminA, "admin_b": self.adminB}
            for key, org, role in (("rev_a", self.orgA, "REVIEWER"), ("rev2_a", self.orgA, "REVIEWER"), ("rev_b", self.orgB, "REVIEWER")):
                self.users[key] = add_user(s, org, f"{key}@example.org", PW, role)
            s.commit()
            self.ids = {k: u.user_id for k, u in self.users.items()}
            self.orgA_id, self.orgB_id = self.orgA.organisation_id, self.orgB.organisation_id

    def evaluate(self, *a, **k):  # noqa: ANN002  (the Evaluator interface: the app calls this)
        ev = getattr(self, "_fixed", None) or ScriptedEvaluator(rulebook=self.rb, **self.script)
        return ev.evaluate(*a, **k)

    def use(self, evaluator) -> None:  # noqa: ANN001
        self._fixed = evaluator

    def login(self, who: str) -> dict:
        email = f"{who.replace('_', '-')}@example.org" if who.startswith("admin") else f"{who}@example.org"
        r = self.client.post("/auth/login", json={"email": email, "password": PW})
        assert r.status_code == 200, r.text
        return {"Authorization": "Bearer " + r.json()["access_token"]}

    def upload(self, who: str = "rev_a", seed: int = 1, name: str = "photo.jpg"):
        return self.client.post("/photos/upload", files={"file": (name, jpeg_bytes(seed), "image/jpeg")}, headers=self.login(who))


    def close(self) -> None:
        self.engine.dispose()
        if self._admin_engine is not None:
            with self._admin_engine.begin() as c:
                c.exec_driver_sql(f"DROP SCHEMA {self._pg_schema} CASCADE")
            self._admin_engine.dispose()


@pytest.fixture()
def P(tmp_path, rb) -> Platform:
    p = Platform(tmp_path, rb)
    yield p
    p.close()


@pytest.fixture()
def PE(tmp_path, rb) -> Platform:
    """A platform whose photographs are embedded (the deterministic thumbnail embedder: no model, no network), for the example store."""
    from ngo_agent.embeddings import ThumbnailEmbedder

    p = Platform(tmp_path, rb, embedder=ThumbnailEmbedder())
    yield p
    p.close()


@pytest.fixture()
def PG(tmp_path, rb) -> Platform:
    """A platform that analyses the gaps between the AI's reasons and the person's after each decision (the offline baseline analyzer: no model)."""
    from ngo_agent.gap_service import BaselineGapAnalyzer

    p = Platform(tmp_path, rb, gap_analyzer=BaselineGapAnalyzer())
    yield p
    p.close()
