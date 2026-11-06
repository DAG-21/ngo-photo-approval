"""A self-contained page (photographs embedded) of the four photographs the staff approved and Haiku 5.5 did not approve in the 20-photograph test of the final rules,
with what Haiku said and what the staff wrote. Reads the saved results and the local demo database (read-only); calls no model.
Output: agent-platform/reports/haiku_misses.html"""
import base64
import html
import io
import json
import sqlite3
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "agent-platform" / "src"), str(ROOT / "layered-engine" / "src")]
from ngo_agent.storage import LocalStorage  # noqa: E402

RESULTS = sorted((ROOT / "agent-platform" / "reports").glob("final_rules_*.jsonl"))[-1]
rows = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line.strip()]
approves = lambda v: v in ("APPROVE", "APPROVED_PENDING_PUBLICATION_CHECKS")  # noqa: E731
haiku = {r["photo"]: r for r in rows if r["config"].startswith("Claude") and r["ok"]}
gemini = {r["photo"]: r for r in rows if r["config"].startswith("Gemini") and r["ok"]}
misses = [n for n, r in haiku.items() if r["staff"] == "APPROVE" and not approves(r["verdict"])]

db = sqlite3.connect(f"file:{(ROOT / 'agent-platform/demo_data/fresh/fresh.db').as_posix()}?mode=ro", uri=True)
store = LocalStorage(ROOT / "agent-platform" / "demo_data" / "fresh" / "storage")
e = html.escape


def load(name):
    row = db.execute("select p.image_storage_path, h.reviewer_comments, h.reason_categories from photos p join human_reviews h on h.photo_id = p.photo_id "
                     "where p.original_filename = ? and h.review_status = 'FEEDBACK_SUBMITTED'", (name,)).fetchone()
    if row:
        return store.get(row[0]), (row[1] or ""), (row[2] or "")
    path = ROOT / "image-classifier" / "data" / "raw" / name          # a labelled photograph from the folders (no staff written reasons exist for these)
    return path.read_bytes(), "", ""


def data_uri(raw):
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((1100, 1100))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


cards = []
for name in misses:
    raw, comments, cats = load(name)
    h, g = haiku[name], gemini.get(name)
    reasons = "".join(f"<li>{e(r)}</li>" for r in h.get("reasons", []))
    staff = f"<p><b>Staff wrote:</b> {e(comments)}</p>" if comments else "<p><b>Staff:</b> approved (no written reasons: this photograph comes from the labelled folders).</p>"
    cards.append(f"""<section><h2>{e(name)}</h2><div class="row"><img src="{data_uri(raw)}" alt="{e(name)}"><div>
<p><b>Staff decision:</b> APPROVE</p>{staff}
<p><b>Haiku 5.5:</b> <span class="bad">{e(h['verdict'])}</span>{' - gates that fired: ' + e(', '.join(h['failed_gates'])) if h['failed_gates'] else ' - no gate fired; the quality score decided'}</p>
<p><b>What Haiku's verdict says:</b></p><ul>{reasons}</ul>
<p class="mute">For comparison, Gemini 2.5 Flash-Lite said {e(g['verdict']) if g else 'nothing usable'}{' (gates ' + e(', '.join(g['failed_gates'])) + ')' if g and g['failed_gates'] else ''}.</p></div></div></section>""")

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Haiku: approved photos it did not approve</title>
<style>:root{{--bg:#fbfbfa;--card:#fff;--ink:#111;--mute:#555;--line:#ddd;--red:#8a1c1c}}@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--card:#1e1e1e;--ink:#eee;--mute:#aaa;--line:#333;--red:#f19a9a}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1040px;margin:0 auto;padding:24px 16px 60px}}h1{{font-size:24px;margin:0 0 6px}}
h2{{font-size:17px;margin:0 0 10px}}section{{background:var(--card);border:1px solid var(--line);padding:14px 16px;margin:16px 0}}.row{{display:flex;gap:18px;flex-wrap:wrap}}img{{max-width:100%;width:480px;height:auto;border:1px solid var(--line)}}
.row>div{{flex:1;min-width:260px}}.bad{{color:var(--red);font-weight:700}}.mute,.lede{{color:var(--mute)}}ul{{margin:4px 0 8px 18px;padding:0}}</style></head><body><main>
<h1>Approved by staff, not approved by Haiku 5.5</h1><p class="lede">The {len(misses)} photographs of the 20-photograph test of the final rules where Haiku disagreed with the staff's approval. Results file: {e(RESULTS.name)}. The quality grades per dimension were not saved in that file, only the verdict and its reasons.</p>
{"".join(cards)}</main></body></html>"""
out = ROOT / "agent-platform" / "reports" / "haiku_misses.html"
out.write_text(page, encoding="utf-8")
print(out, len(misses), "photographs")
