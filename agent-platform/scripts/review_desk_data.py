"""The photographs and AI verdicts for the review desk prototype (review-desk/ in this project): about 50 photographs, each with the AI's verdict and its reasons.

    python scripts/review_desk_data.py                      # dry run: lists the photographs and the estimated cost; NO model call, nothing written
    python scripts/review_desk_data.py --from-demo          # NO model call: only the photographs the local demo has already judged (final rules, Claude Haiku 5.5)
    python scripts/review_desk_data.py --go --max-usd 0.15  # makes the model calls for the rest (Claude Haiku 5.5; the photographs go to Anthropic)

The photographs: the ones the demo has already judged with the final rules and Haiku (no new call for them), topped up to 50 with photographs from the NGO's labelled folders (half approved, half
disapproved by the staff, JPEG only, fixed seed), never the pending-approval folder and never one of the 20 photographs of the earlier comparison. The prompt for a new photograph is the demo's
LIVE one (layered_final_v1 plus the rule rewrites the reviewers' feedback has produced), short answer format. A spending cap (--max-usd) stops the run; a failed call is skipped.

Output, in review-desk/: photos/p01.jpg ... (at most 1400 px on the long side) and data/data.js, which holds for every photograph its AI verdict (APPROVE or DISAPPROVE), the reasons, the answer to every
check and the activity type the AI recognised. It carries no staff decision: those go to reports/review_desk_staff.json here, only to check the AI, and are shown nowhere."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import compare_short_format as csf   # photograph selection and helpers (its main() does not run)

from layered_engine.providers import load_vision_config, make_provider, read_env_file  # noqa: E402

HERE = csf.HERE
FINAL = "layered_final_v1"
TOTAL = 50
SEED = 11
MAX_SIDE = 1400
EST_USD = 0.0021      # per photograph, measured on the last evaluations (cached rules text, short answer with a reason for every check)
OUT = HERE.parent / "review-desk"

NAMES = {
    "R001_POSITION": "Banner is fully in the frame", "R001_VISIBLE": "Banner is visible and readable", "R001_FLAT": "Banner is flat, not crumpled", "R002_GIVER_WHITE": "Giver wears a white top",
    "R003_FACE": "Faces are visible", "R003_FACE_SHARP": "Faces are sharp", "R003_CROP": "Beneficiaries are inside the frame", "R003_OBSTRUCTION_PPE_ONLY": "A covered face is only protective equipment",
    "R004_GENUINE": "Natural, nothing forced (posed is fine)", "R006_ITEM_SHARP": "The item being given is sharp", "R006_ACTION": "A seva activity is taking place",
    "R006_SIZE": "The main people and the activity are clearly visible", "R007_ANGLE": "Camera angle is level and at the people's height", "R007_CUTOFF": "Nothing important is cut off",
    "R007_DISTORTION": "No wide-angle distortion", "R007_PLACEMENT": "Subject is well placed and not blocked", "R008_BACK": "Nobody is shown fully from behind",
    "R010_TECHNICAL": "Focus, exposure and lighting", "R011_BACKGROUND": "Background is not distracting", "R012_DIGNITY": "Everyone's dignity is kept",
    "R013_HUMANITY": "Humanity and compassion come through", "R015_PERSON_PRESENT": "A relevant person is visible", "R015_BEN_RECEIVING": "A beneficiary is receiving the item", "NEW-014": "No watermark or overlay",
}
# the short form shown on a photograph when a mandatory check fails
FAILS = {
    "R001_POSITION": "The banner is cut off by the frame", "R001_VISIBLE": "The banner is not visible enough", "R001_FLAT": "The banner is too crumpled to read", "R002_GIVER_WHITE": "A giver is not in a white top",
    "R003_FACE": "Faces are not visible", "R003_FACE_SHARP": "Faces are not sharp", "R006_ITEM_SHARP": "The item being given is not sharp", "R007_ANGLE": "The camera angle is not level with the people",
    "R008_BACK": "A person is shown fully from behind", "R015_PERSON_PRESENT": "No relevant person is visible", "R015_BEN_RECEIVING": "No beneficiary is receiving the item",
}
AREAS = {"activity_communication": "Activity and subject communication", "technical_quality": "Technical quality", "composition": "Composition and framing",
         "naturalness_storytelling": "Naturalness and storytelling", "brand_presentation": "Brand presentation", "visual_polish": "Visual polish"}
ACTIVITY = {"FOOD_DISTRIBUTION": "Food distribution", "EDUCATION_SUPPORT": "Education support", "HEALTHCARE_ASSISTANCE": "Healthcare", "ANIMAL_CARE": "Animal care", "COMMUNITY_SERVICE": "Community service",
            "DONATION_DISTRIBUTION": "Donation distribution", "AWARENESS_CAMPAIGN": "Awareness campaign", "EVENT_OR_GROUP": "Event or group photo", "BENEFICIARY_PORTRAIT": "Beneficiary portrait",
            "VOLUNTEER_PORTRAIT": "Volunteer portrait", "OTHER": "Other activity"}


def tidy(text: object, limit: int = 160) -> str:
    """One short plain sentence from the AI's own text. One stored answer ran on past its note with stray quotes, braces, a backtick and an apology, so the text is cut at the first character that
    does not belong in a sentence and at a word boundary near `limit`: the review desk never shows such leftovers."""
    import re

    s = re.split(r"[\"{}`<>]", str(text or ""), maxsplit=1)[0]
    s = " ".join(s.split())
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0]
    return s.rstrip(" .;,:")


def describe(decision: dict, validated: dict) -> dict:
    """What the review desk shows for one photograph: the verdict (the AI always gives APPROVE or DISAPPROVE), one headline, the reasons and the answer to every check."""
    approves = csf.approves(decision["recommendation"])
    answers = {c["criterion_id"]: c for c in validated.get("layer_2_criteria") or []}
    clean = tidy
    reasons = []
    for g in decision.get("gates") or []:
        if g.get("hard") and g.get("applicable", True) and g.get("status") in ("NO", "CANNOT_TELL"):
            why = clean(answers.get(g["criterion_id"], {}).get("evidence"))
            head = FAILS.get(g["criterion_id"], NAMES.get(g["criterion_id"], g["criterion_id"]))
            reasons.append({"title": head + (" (could not confirm)" if g["status"] == "CANNOT_TELL" else ""), "why": why})
    q = decision.get("quality") or {}
    dims = validated.get("layer_3_quality_dimensions") or {}
    low = [(k, d) for k, d in dims.items() if d.get("applicable") and d.get("score") is not None and d["score"] <= 2]
    score = q.get("overall_score")
    if not approves and not reasons:
        import re

        sev = str(q.get("severe_weakness") or "")                                   # "POOR:<area>" or "UNUSABLE:<area>": an area graded 1 or 0 disapproves on its own, whatever the total
        cut = re.search(r"cutoff of ([0-9.]+)", " ".join(decision.get("decision_reasons") or []))
        cutoff = float(cut.group(1)) if cut else None
        area = sev.split(":", 1)[1] if ":" in sev else None
        if sev.startswith("UNUSABLE") and area:
            line = f"A part of the photo quality is unusable: {AREAS.get(area, area).lower()}"
        elif sev.startswith("POOR") and area and score is not None and cutoff is not None and score >= cutoff:
            line = f"One part of the photo quality is poor: {AREAS.get(area, area).lower()} (the total, {score:g} out of 100, is above the cutoff of {cutoff:g})"
        else:
            line = f"The overall photo quality is too low ({score:g} out of 100)" if score is not None else "The overall photo quality is too low"
            area = None
        worst = [kd for kd in low if kd[0] == area][:1] or sorted(low, key=lambda kd: kd[1]["score"])[:1]
        reasons.append({"title": line, "why": (f"{AREAS.get(worst[0][0], worst[0][0])}: " + clean(worst[0][1].get("explanation"))).rstrip(": ") if worst else ""})
    checks = [{"name": NAMES.get(cid, cid), "status": a["status"], "why": clean(a.get("evidence"))} for cid, a in answers.items()]
    facts = (validated.get("layer_1_scene_understanding") or {}).get("scene_facts") or {}
    atype = str(facts.get("f_activity_type") or "OTHER")
    return {"verdict": "APPROVE" if approves else "DISAPPROVE", "headline": reasons[0]["title"] if reasons else "All the mandatory checks passed", "reasons": reasons,
            "quality": score, "lowAreas": [{"area": AREAS.get(k, k), "score": d["score"], "why": clean(d.get("explanation"))} for k, d in low], "checks": checks,
            "activity": ACTIVITY.get(atype, atype.replace("_", " ").title()), "summary": clean((validated.get("layer_1_scene_understanding") or {}).get("activity_description"))}


def shrink(data: bytes) -> bytes | None:
    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=84)
            return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


def demo_rows(db: sqlite3.Connection) -> list[dict]:
    """The photographs the demo has already judged with the final rules and Haiku (the latest evaluation of each): no model call."""
    from ngo_agent.storage import LocalStorage

    store = LocalStorage(HERE / "demo_data" / "fresh" / "storage")
    rows = db.execute("""select p.original_filename, p.image_storage_path, p.file_hash, e.decision_output, e.validated_response,
                                (select h.human_final_decision from human_reviews h where h.photo_id = p.photo_id order by h.reviewed_at desc limit 1)
                         from evaluation_runs e join photos p on p.photo_id = e.photo_id
                         where e.evaluation_profile = ? and e.processing_status = 'AI_EVALUATED' and e.model_id like 'claude-haiku%' order by e.created_at""", (FINAL,)).fetchall()
    latest = {r[1]: r for r in rows}
    return [{"name": r[0], "hash": r[2], "bytes": store.get(r[1]), "info": describe(json.loads(r[3]), json.loads(r[4])), "staff": r[5]} for r in latest.values()]


def pick(db: sqlite3.Connection, need: int, skip: set[str]) -> list[dict]:
    from PIL import Image

    demo = csf.demo_photos(db)[:csf.TOTAL]
    earlier = {p["hash"] for p in demo} | {p["name"] for p in demo} | skip
    earlier |= {p["name"] for p in csf.labelled_photos(csf.TOTAL - len(demo), {p["hash"] for p in demo})}      # the labelled ones of the 20-photograph comparison
    rng = random.Random(SEED)
    out: list[dict] = []
    for folder, staff in (("approved", "APPROVE"), ("disapproved", "REJECT")):
        files = sorted((csf.RAW / folder).glob("*"))
        rng.shuffle(files)
        taken = 0
        for p in files:
            if taken >= (need + 1) // 2 if staff == "APPROVE" else taken >= need // 2:
                break
            if f"{folder}/{p.name}" in earlier:
                continue
            data = p.read_bytes()
            if hashlib.sha256(data).hexdigest() in earlier:
                continue
            try:
                with Image.open(io.BytesIO(data)) as im:
                    if im.format != "JPEG":
                        continue                       # the folders hold HEIC and MPO files under .jpg names
                    w, h = im.size
            except Exception:  # noqa: BLE001
                continue
            out.append({"name": f"{folder}/{p.name}", "file": p, "mime": "image/jpeg", "meta": {"width": w, "height": h}, "staff": staff})
            taken += 1
    return out


def write_output(rows: list[dict]) -> None:
    (OUT / "photos").mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    random.Random(SEED + 1).shuffle(rows)       # approved and disapproved are mixed, as in a real queue
    items, staff = [], {}
    for r in rows:
        jpg = shrink(r["bytes"])
        if jpg is None:
            print(f"  skipped (cannot be shown): {r['name']}")
            continue
        pid = f"p{len(items) + 1:02d}"
        (OUT / "photos" / f"{pid}.jpg").write_bytes(jpg)
        items.append({"id": pid, "file": f"photos/{pid}.jpg", "name": r["name"], **r["info"]})      # `name`: the original file name, shown only while the page's SHOW_FILE_NAMES demo switch is on
        staff[pid] = {"photo": r["name"], "staff": r.get("staff"), "ai": r["info"]["verdict"]}
    meta = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "profile": FINAL, "model": "Claude Haiku 5.5", "photos": items}
    (OUT / "data" / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(meta, indent=1, ensure_ascii=False) + ";\n", encoding="utf-8")
    ref = HERE / "reports" / "review_desk_staff.json"
    ref.parent.mkdir(exist_ok=True)
    ref.write_text(json.dumps(staff, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(items)} photographs and data/data.js to {OUT}; staff decisions (shown nowhere) in {ref}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="make the model calls (without it: a dry run, nothing is sent)")
    ap.add_argument("--from-demo", action="store_true", help="no model call: only the photographs the demo has already judged")
    ap.add_argument("--max-usd", type=float, default=0.15, help="stop once the measured spend reaches this (default 0.15 USD, about Rs 14)")
    args = ap.parse_args()

    db = sqlite3.connect(f"file:{csf.DB.as_posix()}?mode=ro", uri=True)
    have = demo_rows(db)
    if args.from_demo:
        print(f"{len(have)} photographs already judged on the demo (final rules, Claude Haiku 5.5); no model call")
        write_output(have)
        return 0

    photos = pick(db, TOTAL - len(have), {r["hash"] for r in have})
    print(f"{len(have)} photographs already judged on the demo (no new call) + {len(photos)} new ones ({sum(p['staff'] == 'APPROVE' for p in photos)} approved and "
          f"{sum(p['staff'] == 'REJECT' for p in photos)} disapproved by staff; none pending, none of the earlier 20):")
    for p in photos:
        print(f"   {p['name']}")
    est = EST_USD * len(photos)
    print(f"Estimated ${est:.3f} (about Rs {est * csf.INR_PER_USD:.1f}) with Claude Haiku 5.5: {len(photos)} photographs go to Anthropic. Spending cap ${args.max_usd:.2f}.")
    if not args.go:
        print("Dry run: nothing was sent and nothing was written. Add --go to make the calls.")
        return 0

    from ngo_agent.evaluators import LayeredEngineEvaluator

    live = csf.live_prompt(db)               # the rules the demo is using now, including the rewrites from the reviewers' feedback
    live["profile"] = FINAL
    cfg = load_vision_config("haiku")
    env = read_env_file(HERE.parent / "image-classifier" / ".env", (cfg["auth"]["api_key_env"], cfg["auth"]["authorization_env"]))
    ev = LayeredEngineEvaluator(make_provider(cfg, env=env), cfg, provider_name="anthropic-claude")   # fails closed without the switch and the key
    spent, rows = 0.0, list(have)
    for p in photos:
        if spent >= args.max_usd:
            print(f"spending cap reached (${spent:.3f}): stopping")
            break
        data = p["file"].read_bytes()
        started = time.monotonic()
        try:
            r = ev.evaluate(data, p["mime"], live, p["meta"])
            spent += r.cost_usd or 0.0
            info = describe(r.decision, r.validated)
            rows.append({"name": p["name"], "bytes": data, "info": info, "staff": p["staff"]})
            print(f"  {p['name']:34} {info['verdict']:10} ${r.cost_usd or 0:.4f} {time.monotonic() - started:3.0f}s  {info['headline'][:60]}", flush=True)
        except Exception as exc:  # noqa: BLE001  (a rejected request is not billed)
            print(f"  {p['name']:34} ERROR {type(exc).__name__}: {str(exc)[:150]}", flush=True)
    write_output(rows)
    agree = sum((r["info"]["verdict"] == "APPROVE") == (r["staff"] == "APPROVE") for r in rows if r.get("staff"))
    judged = sum(1 for r in rows if r.get("staff"))
    false_app = sum(r["info"]["verdict"] == "APPROVE" and r["staff"] == "REJECT" for r in rows)
    print(f"\nspent ${spent:.3f} (about Rs {spent * csf.INR_PER_USD:.1f}); {len(rows)} photographs; agrees with staff {agree}/{judged}; FALSE APPROVALS {false_app}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
