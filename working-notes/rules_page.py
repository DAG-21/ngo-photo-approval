"""Write a readable page of every rule the AI is given, with the wording in force on the local demo (rewrites included). Reads only; changes nothing."""
import html
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # the project folder (this script lives in working-notes/)
sys.path[:0] = [str(ROOT / "agent-platform" / "src"), str(ROOT / "layered-engine" / "src")]
from layered_engine.prompt_builder import FACT_FIELDS, guidance_of  # noqa: E402
from layered_engine.rulebook import Rulebook  # noqa: E402
from ngo_agent.rule_drafter import rule_catalogue  # noqa: E402

_c = sqlite3.connect(f"file:{(ROOT / 'agent-platform/demo_data/fresh/fresh.db').as_posix()}?mode=ro", uri=True)
P = json.loads(_c.execute("select configuration from rulebook_versions where version_id=(select active_rulebook_version_id from organisations)").fetchone()[0])["profile"]   # the profile the demo uses now
_c.close()
rb = Rulebook()
prof = rb.resolve(P)
cat = rule_catalogue(rb, P)

db = sqlite3.connect(f"file:{(ROOT / 'agent-platform/demo_data/fresh/fresh.db').as_posix()}?mode=ro", uri=True)
pv_id = db.execute("select active_prompt_version_id from organisations").fetchone()[0]
live_text = json.loads(db.execute("select configuration from prompt_versions where version_id=?", (pv_id,)).fetchone()[0]).get("rule_text") or {}
last_status = {}
for edit_id, status in db.execute("select edit_id, status from rule_edit_events order by edit_id, seq"):
    last_status[edit_id] = status
live_edit = {}
for edit_id, target, created, rationale, source in db.execute("select edit_id, target_id, created_at, rationale, source from rule_edits order by created_at"):
    if last_status.get(edit_id) in ("APPLIED", "CONFIRMED"):
        live_edit[target] = {"when": created[:16], "why": rationale, "admin": (json.loads(source or "{}").get("administrator") or {}).get("by")}

e = html.escape
SRC = {"CLIENT": "client's rule", "TEAM": "added by the team", "PROPOSED": "proposed, awaiting NGO approval"}


def effect(cid):
    i = cat[cid]
    c = prof.criteria.get(cid)
    if i.gate_id and i.hard:
        return ("rejects", f"Gate {i.gate_id}: can reject the photo")
    if i.gate_id:
        return ("soft", f"Gate {i.gate_id} (soft): lowers a quality score")
    if c is not None and c.decision_effect == "SOFT_SCORE":
        return ("soft", "Lowers a quality score")
    return ("info", "Information only")


n = 0
rows_out = []


LIST = []


def rule_row(cid, label_html):
    global n
    n += 1
    i = cat[cid]
    import re as _re
    LIST.append({"n": n, "id": cid, "label": _re.sub(r"<[^>]+>", " | ", label_html).strip(" |").replace(" |  | ", " | "), "base": i.base, "now": live_text.get(cid)})
    now = live_text.get(cid)
    changed = now is not None and " ".join(now.split()) != " ".join(i.base.split())
    ed = live_edit.get(cid)
    who = f"set by an administrator ({e(ed['admin'])})" if ed and ed["admin"] else "rewritten by the learning loop from reviewers' feedback"
    body = (f'<div class="now">{e(now if changed else i.base)}</div>'
            + (f'<div class="was"><b>Original:</b> {e(i.base)}</div><div class="why">Changed {e(ed["when"]) if ed else ""}, {who}{": " + e(ed["why"]) if ed and ed["why"] else ""}</div>' if changed else ""))
    return f'<li class="rule{" changed" if changed else ""}" id="r{n}" data-text="{e((cid + " " + (now or i.base)).lower())}"><div class="head"><span class="num">#{n}</span><code>{e(cid)}</code>{label_html}{"<span class=tag-chg>changed</span>" if changed else ""}</div>{body}</li>'


sections = []
asked = prof.asked_criteria()
for r in rb.sources["client_rules"]:
    rid = r["rule_id"]
    items = [rule_row(rid, '<span class="tag client">client\'s own text (shown to the AI for reference)</span>')]
    for c in sorted([c for c in asked if c.parent_rule_id == rid], key=lambda c: (c.priority, c.criterion_id)):
        k, txt = effect(c.criterion_id)
        items.append(rule_row(c.criterion_id, f'<span class="tag {k}">{e(txt)}</span><span class="tag src">{e(SRC.get(c.source, c.source))}</span>'))
    sections.append(f'<section><h2>{e(rid)} · {e(r.get("source_ref", ""))}</h2><ol>{"".join(items)}</ol></section>')
other = [c for c in asked if not c.parent_rule_id or c.parent_rule_id not in {r["rule_id"] for r in rb.sources["client_rules"]}]
if other:
    items = []
    for c in sorted(other, key=lambda c: (c.priority, c.criterion_id)):
        k, txt = effect(c.criterion_id)
        items.append(rule_row(c.criterion_id, f'<span class="tag {k}">{e(txt)}</span><span class="tag src">{e(SRC.get(c.source, c.source))}</span>'))
    sections.append(f'<section><h2>Other questions</h2><ol>{"".join(items)}</ol></section>')
facts = sorted([c for c in prof.criteria.values() if c.decision_effect == "FACT" and c.criterion_id in FACT_FIELDS], key=lambda c: c.criterion_id)
sections.append('<section><h2>Scene facts (what the AI reports about the scene first; Python uses them to decide which rules apply)</h2><ol>'
                + "".join(rule_row(c.criterion_id, '<span class="tag info">scene fact</span>') for c in facts) + "</ol></section>")
sections.append('<section><h2>Team guidance (general statements given to the AI with the rules)</h2><ol>'
                + "".join(rule_row(g, '<span class="tag guide">team guidance</span>') for g in guidance_of(prof)) + "</ol></section>")

hard = sum(1 for g in prof.gates if prof.is_hard(g))
changed_n = sum(1 for t, v in live_text.items() if t in cat and " ".join(v.split()) != " ".join(cat[t].base.split()))
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All rules</title>
<style>
:root{{--bg:#fbfbfa;--card:#fff;--ink:#111;--mute:#555;--line:#ddd;--hl:#fff3c4;--red:#8a1c1c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--card:#1e1e1e;--ink:#eee;--mute:#aaa;--line:#333;--hl:#3a3315;--red:#f19a9a}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px 80px}} h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:17px;margin:28px 0 8px;border-bottom:2px solid var(--ink);padding-bottom:4px}}
.lede{{color:var(--mute);margin:0 0 14px}} .bar{{position:sticky;top:0;background:var(--bg);padding:10px 0;z-index:2}}
input{{width:100%;box-sizing:border-box;padding:10px 12px;font:inherit;border:1.5px solid var(--ink);background:var(--card);color:var(--ink)}}
ol{{list-style:none;margin:0;padding:0}} .rule{{background:var(--card);border:1px solid var(--line);padding:10px 14px;margin:8px 0}} .rule.changed{{border-left:5px solid var(--ink);background:var(--hl)}}
.head{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:4px}} .num{{font-weight:700;min-width:44px}} code{{font:13px ui-monospace,Consolas,monospace}}
.tag{{font-size:12px;padding:1px 7px;border:1px solid var(--line);color:var(--mute)}} .tag.rejects{{border-color:var(--red);color:var(--red);font-weight:700}} .tag-chg{{font-size:12px;padding:1px 7px;background:var(--ink);color:var(--bg);font-weight:700}}
.was,.why{{font-size:13px;color:var(--mute);margin-top:4px}} .stats{{display:flex;flex-wrap:wrap;gap:14px;font-size:14px;color:var(--mute)}}
</style></head><body><main>
<h1>All rules the AI is given</h1><p class="lede">Profile in use: <b>{P}</b></p>
<p class="lede">The wording in force on the local demo now, rewrites included. Refer to any rule by its number or id when you want it changed.</p>
<div class="stats"><span><b>{n}</b> rules</span><span><b>{len(asked)}</b> questions</span><span><b>{len(facts)}</b> scene facts</span><span><b>16</b> client rules</span><span><b>{len(guidance_of(prof))}</b> guidance lines</span><span><b>{hard}</b> gates that can reject, <b>{len(prof.gates) - hard}</b> soft</span><span><b>{changed_n}</b> changed from the rulebook</span></div>
<div class="bar"><input id="q" type="search" placeholder="Filter: type words or an id, e.g. banner, gaze, R004, #42"></div>
{"".join(sections)}
</main><script>
const q=document.getElementById('q');q.addEventListener('input',()=>{{const t=q.value.trim().toLowerCase();document.querySelectorAll('.rule').forEach(li=>{{const num='#'+li.id.slice(1);li.style.display=!t||li.dataset.text.includes(t)||num===t?'':'none';}});}});
</script></body></html>"""
out = ROOT / "agent-platform" / "reports" / "rules_current.html"
out.parent.mkdir(exist_ok=True)
out.write_text(page, encoding="utf-8")
(ROOT / "agent-platform" / "reports" / "rules_current.json").write_text(json.dumps(LIST, ensure_ascii=False, indent=1), encoding="utf-8")
print(out, "| rules:", n, "| changed:", changed_n, "| gates hard/soft:", hard, len(prof.gates) - hard)
