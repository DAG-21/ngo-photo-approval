"""Phase 2A demonstration on REAL stored evaluations (no model call, no network): scenarios A to G of the specification.

    python scripts/demo_phase2a.py [--serve]

It builds a local demo database under demo_data/ (git-ignored: it holds client photographs), imports a few photographs from pilot 3 together with
the answers the engine already produced for them (layered-engine/runs/pilot3_layered), and walks through the seven scenarios through the real
HTTP API. With --serve it then starts the review interface on http://127.0.0.1:8100 using the same data. Demo passwords are random and are
written to demo_data/credentials.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / "src"), str(HERE.parent / "layered-engine" / "src")]

from fastapi.testclient import TestClient  # noqa: E402

from layered_engine.annotation import read_rows  # noqa: E402
from ngo_agent.api.app import create_app  # noqa: E402
from ngo_agent.bootstrap import add_user, create_organisation  # noqa: E402
from ngo_agent.db import create_schema, make_engine, make_session_factory  # noqa: E402
from ngo_agent.embeddings import make_embedder  # noqa: E402
from ngo_agent.gap_service import BaselineGapAnalyzer  # noqa: E402
from ngo_agent.evaluators import EvaluatorOutput, ScriptedEvaluator  # noqa: E402
from layered_engine.rulebook import Rulebook  # noqa: E402
from ngo_agent.storage import LocalStorage, sha256_hex  # noqa: E402

ENGINE = HERE.parent / "layered-engine"
DEMO = HERE / "demo_data"
PICK = ["NGO_IMG_000202", "NGO_IMG_000203", "NGO_IMG_000205", "NGO_IMG_000210"]   # a false approval, an approved photo auto-approved, a rejected approved photo, one more


class StoredEvaluator:
    """Returns the answers the engine already produced for a photograph (looked up by file hash). Nothing is sent anywhere."""

    def __init__(self) -> None:
        paths = {r["image_id"]: Path(r["path"]) for r in csv.DictReader((ENGINE / "pilot" / "list3.csv").open(encoding="utf-8"))}
        rows = {r["image_id"]: r for r in read_rows(ENGINE / "runs" / "pilot3_layered" / "results.jsonl") if r.get("ok")}
        self.by_hash = {sha256_hex(paths[i].read_bytes()): rows[i] for i in rows if paths[i].exists()}
        self.paths = paths
        self.script: dict[str, dict] = {}      # synthetic photographs of scenario I: file hash -> the scripted answers (no model, no real photograph)
        self.rb = Rulebook()

    def evaluate(self, image_bytes, mime_type, rulebook_config, metadata, examples=None):  # noqa: ANN001
        self.last_examples = examples          # the examples the service would show a real model; this stand-in only records that it was given them
        if sha256_hex(image_bytes) in self.script:
            return ScriptedEvaluator(rulebook=self.rb, **self.script[sha256_hex(image_bytes)]).evaluate(image_bytes, mime_type, rulebook_config, metadata, examples=examples)
        row = self.by_hash[sha256_hex(image_bytes)]
        return EvaluatorOutput(raw_text=row["raw_text"], validated=row["validated"], decision=row["decision"], model_provider="google-gemini (stored result)", model_id=row["model_id"],
                               profile=row["profile"], prompt_hash=row["prompt_hash"], rulebook_version=row["rulebook_version"], usage=row["usage"], cost_usd=row.get("cost_usd"))


def build(reset: bool = True):
    DEMO.mkdir(exist_ok=True)
    db = DEMO / "demo.db"
    if reset and db.exists():
        db.unlink()
    engine = make_engine(f"sqlite:///{db.as_posix()}")
    create_schema(engine)
    factory = make_session_factory(engine)
    pw = {k: secrets.token_urlsafe(12) for k in ("admin_a", "reviewer_a", "reviewer2_a", "admin_b", "reviewer_b", "admin_c", "reviewer_c")}
    with factory() as s:
        orgA, adminA = create_organisation(s, "Demo NGO A", "admin.a@demo.local", pw["admin_a"])
        orgB, adminB = create_organisation(s, "Demo NGO B", "admin.b@demo.local", pw["admin_b"])
        orgC, adminC = create_organisation(s, "Demo NGO C (synthetic photographs, learning loop)", "admin.c@demo.local", pw["admin_c"])
        for email, key, org, role in (("reviewer.a@demo.local", "reviewer_a", orgA, "REVIEWER"), ("reviewer2.a@demo.local", "reviewer2_a", orgA, "REVIEWER"),
                                      ("reviewer.b@demo.local", "reviewer_b", orgB, "REVIEWER"), ("reviewer.c@demo.local", "reviewer_c", orgC, "REVIEWER")):
            add_user(s, org, email, pw[key], role)
        s.commit()
        ids = (orgA.organisation_id, orgB.organisation_id, orgC.organisation_id)
    (DEMO / "credentials.txt").write_text("\n".join(f"{k}: {v}" for k, v in pw.items()) + "\n(local demo only; emails are <role>.<a|b|c>@demo.local)\n", encoding="utf-8")
    ev = StoredEvaluator()
    storage = LocalStorage(DEMO / "storage")
    return engine, factory, storage, ev, pw, ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()
    engine, factory, storage, ev, pw, (orgA, orgB, orgC) = build()
    emb = make_embedder(os.environ.get("DEMO_EMBEDDER", "clip"))   # a local model: nothing is downloaded or sent anywhere
    app = create_app(factory, storage, ev, "demo-secret-" + secrets.token_hex(20), run_inline=True, embedder=emb, gap_analyzer=BaselineGapAnalyzer())
    c = TestClient(app)
    login = lambda email, key: {"Authorization": "Bearer " + c.post("/auth/login", json={"email": email, "password": pw[key]}).json()["access_token"]}  # noqa: E731
    rev, rev2, adm = login("reviewer.a@demo.local", "reviewer_a"), login("reviewer2.a@demo.local", "reviewer2_a"), login("admin.a@demo.local", "admin_a")
    revB = login("reviewer.b@demo.local", "reviewer_b")

    def up(oid, headers):
        r = c.post("/photos/upload", files={"file": (f"{oid}.jpg", ev.paths[oid].read_bytes(), "image/jpeg")}, headers=headers)
        assert r.status_code in (200, 201), r.text
        return r.json()

    def show(title):
        print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)

    ups = {oid: up(oid, rev) for oid in PICK}
    print("Uploaded", len(ups), "photographs; each now has an immutable AI evaluation (stored answers from pilot 3, V1 profile).")
    for oid, u in ups.items():
        e = c.get(f"/evaluations/{u['evaluation_id']}", headers=rev).json()
        print(f"  {oid}: AI recommendation {e['ai_recommendation']}, quality {e['quality_score']}, failed gates {e['decision']['failed_gates']}, integrity {e['integrity_ok']}")

    show("Scenario A: the human agrees with the AI (a reason is still required)")
    a = next(o for o in PICK if c.get(f"/evaluations/{ups[o]['evaluation_id']}", headers=rev).json()["ai_recommendation"].startswith("APPROVE"))
    ea = ups[a]["evaluation_id"]
    before = c.get(f"/evaluations/{ea}", headers=rev).json()["content_hash"]
    rid = c.post(f"/evaluations/{ea}/review", headers=rev).json()["review_id"]
    print("  agreeing without a reason:", c.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE"}, headers=rev).json()["detail"][:90] + "...")
    r = c.post(f"/feedback/{rid}/submit", json={"decision": "APPROVE", "comment": "Clear faces, a visible handover and a readable banner."}, headers={**rev, "Idempotency-Key": rid}).json()["review"]
    print(f"  AI said {before and c.get(f'/evaluations/{ea}', headers=rev).json()['ai_recommendation']}; human decided {r['human_final_decision']}; agreement {r['overall_agreement_with_ai']}")
    print("  original evaluation hash unchanged:", c.get(f"/evaluations/{ea}", headers=rev).json()["content_hash"] == before)

    show("Scenario B: the human disregards the AI (REJECT -> APPROVE) and must write why")
    b = next(o for o in PICK if c.get(f"/evaluations/{ups[o]['evaluation_id']}", headers=rev).json()["ai_recommendation"] == "REJECT")
    eb = ups[b]["evaluation_id"]
    rid_b = c.post(f"/evaluations/{eb}/review", headers=rev).json()["review_id"]
    print("  without a reason:", c.post(f"/feedback/{rid_b}/submit", json={"decision": "APPROVE"}, headers=rev).json()["detail"])
    print("  a ticked category alone:", c.post(f"/feedback/{rid_b}/submit", json={"decision": "APPROVE", "reason_categories": ["OTHER"]}, headers=rev).json()["detail"][:60] + "...")

    show("Scenario C: the human corrects one gate (the original is not touched)")
    ev_b = c.get(f"/evaluations/{eb}", headers=rev).json()
    gate = next(g for g in ev_b["layer_2"]["gates"] if g["status"] == "NO" and g["hard"])
    print(f"  AI: {gate['gate_id']} = {gate['status']} - {gate['evidence']}")
    corr = c.patch(f"/evaluations/{eb}/corrections", json={"corrections": [{"criterion_type": "MANDATORY_GATE", "criterion_id": gate["gate_id"], "human_corrected_value": "YES",
                                                                            "correction_reason": ["RULE_APPLICABILITY_ERROR"], "reviewer_comment": "The person is in the background and is not a featured beneficiary."}]}, headers=rev).json()["corrections"][0]
    print(f"  stored correction: AI {corr['ai_predicted_value']} -> human {corr['human_corrected_value']} (revision {corr['revision']})")
    print("  evaluation still says:", next(g for g in c.get(f"/evaluations/{eb}", headers=rev).json()["layer_2"]["gates"] if g["gate_id"] == gate["gate_id"])["status"])

    show("Scenario D: the human adds a tag the AI missed")
    tag = c.patch(f"/evaluations/{eb}/corrections", json={"corrections": [{"criterion_type": "TAG_ADDED", "criterion_id": "cluttered foreground", "human_corrected_value": "PRESENT",
                                                                           "correction_reason": ["VISUAL_PERCEPTION_ERROR"]}]}, headers=rev).json()["corrections"][0]
    print(f"  recorded: {tag['criterion_type']} '{tag['criterion_id']}' (AI value {tag['ai_predicted_value']})")
    why = "The person is in the background and is not a featured beneficiary."
    sub = c.post(f"/feedback/{rid_b}/submit", json={"decision": "APPROVE", "comment": why}, headers=rev).json()["review"]
    print(f"  submitted: AI {ev_b['ai_recommendation']} vs human {sub['human_final_decision']} -> {sub['overall_agreement_with_ai']}; reason on record: \"{sub['reviewer_comments']}\"")
    print("  three separate fields, nothing overwritten")

    show("Scenario E: the person's submission is final and becomes a learning example (inactive until Phase 2C)")
    review = c.get(f"/feedback/{rid_b}", headers=rev).json()
    sent = next(e for e in review["events"] if e["event_type"] == "FEEDBACK_SUBMITTED")["payload"]
    print(f"  review status {review['review_status']}; no validation step; learning example {str(sent['learning_example_id'])[:8]}")
    ex = c.get(f"/organisations/{orgA}/learning-examples", headers=adm).json()["learning_examples"]
    print(f"  learning examples for this NGO: {len(ex)} (active_for_retrieval={[e['active_for_retrieval'] for e in ex]})")
    for g in c.get(f"/feedback/{rid_b}", headers=rev).json()["gap_analysis"]:
        print(f"  gap analysis ({g['analyzer']}, text only, no model): {g['summary']}", *[f"\n    - {x['kind']} {x['related_gates'] or ''}: {x['summary'][:80]}" for x in g["gaps"]])
    print("  the decision cannot be changed afterwards:", c.post(f"/feedback/{rid_b}/submit", json={"decision": "REJECT", "comment": "changed my mind about this"}, headers=rev).status_code, "(409 = refused)")

    show("Scenario F: re-evaluation creates a new run; the earlier one is unchanged")
    pid = ups[b]["photo"]["photo_id"]
    vers = c.get(f"/organisations/{orgA}/versions", headers=adm).json()
    v2r = next(v for v in vers["rulebook_versions"] if v["configuration"]["profile"] == "layered_proposed_v2")
    print(f"  V2 rulebook status: {v2r['approval_status']} (cannot be activated); an administrator may still run it as an EXPERIMENT:")
    print("  reviewer re-evaluates (same versions) ->", c.post(f"/photos/{pid}/evaluations", json={}, headers=rev).status_code, "(stored answer reused by the demo evaluator)")
    photo = c.get(f"/photos/{pid}", headers=rev).json()
    print(f"  evaluations on this photo: {len(photo['evaluations'])}; photo status {photo['photo']['photo_status']} (the person's decision stands)")
    print("  first evaluation hash still intact:", c.get(f"/evaluations/{eb}", headers=rev).json()["integrity_ok"])
    print("  lifecycle:", " > ".join(e["stage"] for e in photo["lifecycle"]))

    show("Scenario G: two organisations are isolated")
    upB = c.post("/photos/upload", files={"file": ("b.jpg", ev.paths[PICK[0]].read_bytes(), "image/jpeg")}, headers=revB).json()
    print("  organisation B uploads the same bytes: its own photo", upB["photo"]["photo_id"][:8], "!=", ups[PICK[0]]["photo"]["photo_id"][:8])
    for url in (f"/photos/{ups[PICK[0]]['photo']['photo_id']}", f"/evaluations/{eb}", f"/feedback/{rid_b}", f"/organisations/{orgA}/feedback"):
        print(f"  B asks for {url.split('/')[1]}/...: HTTP {c.get(url, headers=revB).status_code}")
    print("  B's own feedback list:", c.get(f"/organisations/{orgB}/feedback", headers=revB).json()["reviews"])

    show("Scenario H: similar photographs that staff decided earlier are found and would be shown to the model (no model is called here)")
    labels = {r["opaque_id"]: r["label"] for r in csv.DictReader((ENGINE / "pilot" / "key3_PRIVATE.csv").open(encoding="utf-8"))}   # the pilot's staff decisions, used locally only
    avail = [i for i in sorted(ev.paths) if i not in PICK and i in labels and ev.paths[i].exists() and sha256_hex(ev.paths[i].read_bytes()) in ev.by_hash]
    appr, dis = [i for i in avail if labels[i] == "approved"], [i for i in avail if labels[i] != "approved"]
    k = min(5, len(appr) - 1, len(dis) - 1)                                              # as many as the stored pilot results allow; one of each class is held back as a new photograph
    pool, queries = appr[:k] + dis[:k], [appr[k], dis[k]]
    print(f"  {len(pool)} past decisions ({k} approved, {k} rejected) and {len(queries)} new photographs")
    on = c.put(f"/organisations/{orgA}/few-shot", json={"enabled": True, "confirm_cost_increase": True}, headers=adm)
    print("  switching the feature on:", on.status_code, on.json() if on.status_code != 200 else "(confirmed: three more photographs in each evaluation, about 8% more cost)")
    for oid in pool:                                                                      # the staff's pilot decisions, entered as reviews with a stated reason
        u = up(oid, rev)
        rid_p = c.post(f"/evaluations/{u['evaluation_id']}/review", headers=rev).json()["review_id"]
        done = c.post(f"/feedback/{rid_p}/submit", json={"decision": "APPROVE" if labels[oid] == "approved" else "REJECT",
                                                         "comment": "Demo: the staff's decision in the pilot (their written reason was not recorded)."}, headers=rev)
        assert done.status_code == 200, done.text
    info = c.get(f"/organisations/{orgA}/few-shot", headers=adm).json()
    print(f"  embedder {info['embedder']}: {info['photos_embedded']} photographs embedded, {info['examples_available']} past decisions available as examples")
    for oid in queries:
        u = up(oid, rev)
        shown = (c.get(f"/evaluations/{u['evaluation_id']}", headers=rev).json().get("few_shot") or {}).get("examples", [])
        row = ", ".join(f"{e['decision'].lower()} {e['similarity']:.0%}" for e in reversed(shown)) or "none"
        print(f"  new photograph {oid} (staff decision: {labels[oid]}): the model would be shown {len(shown)} examples, most similar first: {row}")
    print("  the evaluator was given them (and the screen lists them): ", bool(getattr(ev, "last_examples", None)))

    show("Scenario I: the learning loop on synthetic photographs (a separate demo NGO: the 50-photo pilot is too small to make it act)")
    import io

    from PIL import Image

    from ngo_agent import loop_service as loop

    admC, revC = login("admin.c@demo.local", "admin_c"), login("reviewer.c@demo.local", "reviewer_c")
    held: set[str] = set()
    loop.is_held_out = lambda pid, pct: pid in held        # demo only: a pinned held-out split so that this story is the same every time it is run
    counter = iter(range(1, 10_000))

    def syn(human, answers=None, hold=False, dims=None):
        """One synthetic photograph (a plain coloured image) whose scripted answers are the AI's, decided by `human`."""
        n = next(counter)
        buf = io.BytesIO()
        Image.new("RGB", (120 + n, 90), (40 + n * 7 % 180, 100, 70)).save(buf, "JPEG")
        data = buf.getvalue()
        ev.script[sha256_hex(data)] = {"answers": answers or {}, "dims": dims or {}}
        u = c.post("/photos/upload", files={"file": (f"SYNTHETIC_{n:03d}.jpg", data, "image/jpeg")}, headers=revC).json()
        rid_s = c.post(f"/evaluations/{u['evaluation_id']}/review", headers=revC).json()["review_id"]
        why = "The faces are clear, the handover is visible and the banner is readable." if human == "APPROVE" else "The faces are hidden behind the volunteers and the banner is cropped."
        assert c.post(f"/feedback/{rid_s}/submit", json={"decision": human, "comment": why}, headers=revC).status_code == 200
        if hold:
            held.add(u["photo"]["photo_id"])

    G4, G45 = {"R003_FACE": "NO"}, {"R003_FACE": "NO", "R003_CLEAR": "NO"}
    for _ in range(2):
        syn("APPROVE", G4, hold=True)
    for _ in range(3):
        syn("APPROVE", G4)
    for _ in range(3):
        syn("APPROVE")
    for _ in range(2):
        syn("REJECT", G45, hold=True)
    for _ in range(2):
        syn("REJECT", G45)
    run_c = lambda: c.post(f"/organisations/{orgC}/learning-loop/run", headers=admC).json()["status"]  # noqa: E731
    on = c.put(f"/organisations/{orgC}/learning-loop", json={"enabled": True, "confirm_automatic_changes": True, "every_n": 1000, "min_rows": 6, "monitor_min": 2, "monitor_window": 4, "error_margin": 99}, headers=admC)
    print(f"  12 staff decisions on synthetic photographs; the loop is switched on ({on.status_code}) with the evidence rule it ships with (20 staff-rejected held-out photographs)")
    print("  first cycle:", run_c(), "(the draft, softening gate G004, is held for developers: too few rejected photographs to prove it safe)")
    c.put(f"/organisations/{orgC}/learning-loop", json={"min_rejected_for_loosening": 1}, headers=admC)
    print("  an administrator lowers the evidence minimum to 1 (demo only) and runs a cycle:", run_c())
    for _ in range(3):
        syn("REJECT", G4)                                  # under the new settings the AI approves these, and staff reject them
    print("  three more photographs the AI now approves and staff reject; next cycle:", run_c(), "(the change is rolled back by itself)")
    BORDER = {"activity_communication": 2, "technical_quality": 2, "composition": 3, "naturalness_storytelling": 2, "brand_presentation": 3, "visual_polish": 2}
    for _ in range(2):
        syn("APPROVE", dims=BORDER, hold=True)             # photographs whose quality score sends them to review although staff approve them
    for _ in range(4):
        syn("APPROVE", dims=BORDER)
    print("  six photographs staff approve although the quality score sends them to review; next cycle:", run_c(), "(the review threshold is lowered)")
    c.put(f"/organisations/{orgC}/learning-loop", json={"min_rejected_for_loosening": 20}, headers=admC)
    for _ in range(2):
        syn("APPROVE", {"R003_CLEAR": "NO"}, hold=True)
    for _ in range(3):
        syn("APPROVE", {"R003_CLEAR": "NO"})
    print("  the evidence rule is back to 20 and a new pattern appears (gate G005 alone rejects photographs staff approve); next cycle:", run_c(), "(held for developers)")
    for ch in c.get(f"/organisations/{orgC}/changelog", headers=admC).json()["changes"]:
        print(f"  {ch['label']} {ch['kind']} {ch['target'] or ''}: {ch['status']}; " + " > ".join(h["status"] for h in ch["history"]))

    show("Database-level protection")
    from sqlalchemy import text
    with engine.begin() as conn:
        try:
            conn.execute(text("UPDATE evaluation_runs SET ai_recommendation='APPROVE'"))
        except Exception as exc:  # noqa: BLE001
            print("  UPDATE evaluation_runs ->", str(exc.orig)[:80])
    print("\nDemo database:", DEMO / "demo.db", "| credentials:", DEMO / "credentials.txt")
    if args.serve:
        import uvicorn

        secret = "demo-secret-" + secrets.token_hex(20)
        print("Serving the review interface on http://127.0.0.1:8100 (Ctrl+C to stop)")
        uvicorn.run(create_app(factory, storage, ev, secret, run_inline=False, embedder=emb, gap_analyzer=BaselineGapAnalyzer()), host="127.0.0.1", port=8100, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
