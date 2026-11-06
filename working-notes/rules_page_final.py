"""Write the rules sheet of the profile layered_final_v1 (the owner's rule review of 2026-11-05), compared with layered_tuned_v1 (what the demo uses today).
Reads only the rulebook files and agreed_drafts.json; it does not touch the demo database and calls no model. Output: agent-platform/reports/rules_final.html and rules_final.json."""
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # the project folder (this script lives in working-notes/)
sys.path[:0] = [str(ROOT / "agent-platform" / "src"), str(ROOT / "layered-engine" / "src")]
from layered_engine.prompt_builder import FACT_FIELDS, guidance_of  # noqa: E402
from layered_engine.rulebook import Rulebook  # noqa: E402
from ngo_agent.rule_drafter import rule_catalogue  # noqa: E402

FINAL, OLD = "layered_final_v1", "layered_tuned_v1"
rb = Rulebook()
prof, old_prof = rb.resolve(FINAL), rb.resolve(OLD)
cat, old_cat = rule_catalogue(rb, FINAL), rule_catalogue(rb, OLD)
removals = json.loads((ROOT / "working-notes" / "agreed_drafts.json").read_text(encoding="utf-8")).get("removals", {})
e = html.escape
norm = lambda t: " ".join(str(t or "").split())  # noqa: E731
SRC = {"CLIENT": "client's rule", "TEAM": "added by the team", "PROPOSED": "proposed, awaiting NGO approval"}


def effect(cid):
    i = cat[cid]
    c = prof.criteria.get(cid)
    if i.gate_id and i.hard:
        return ("rejects", f"Gate {i.gate_id}: can reject the photo")
    if i.gate_id:
        return ("soft", f"Gate {i.gate_id} (soft): lowers a quality score, the grade gives the degree")
    if c is not None and c.decision_effect == "SOFT_SCORE":
        return ("soft", "Lowers a quality score")
    return ("info", "Information only")


def old_effect(cid):
    if cid not in old_cat:
        return None
    i = old_cat[cid]
    if i.gate_id and i.hard:
        return f"gate {i.gate_id} (rejects)"
    if i.gate_id:
        return f"gate {i.gate_id} (soft)"
    return "information only" if i.kind == "RULE_QUESTION" else None


n, LIST = 0, []


def rule_row(cid, label_html):
    global n
    n += 1
    i = cat[cid]
    new = cid not in old_cat
    was = None if new else old_cat[cid].base
    changed = not new and norm(was) != norm(i.base)
    role_changed = False
    if not new and i.kind == "RULE_QUESTION":
        eff_now, eff_old = effect(cid)[1].split(":")[0].lower(), old_effect(cid)
        role_changed = bool(eff_old) and (("rejects" in effect(cid)[0]) != ("rejects" in (eff_old or ""))) or (("soft" in effect(cid)[0]) != ("soft" in (eff_old or "")))
    LIST.append({"n": n, "id": cid, "label": re.sub(r"<[^>]+>", " | ", label_html).strip(" |").replace(" |  | ", " | "), "now": i.base, "was_in_tuned": None if new else was, "new": new})
    tag = "<span class=tag-new>new</span>" if new else ("<span class=tag-chg>changed</span>" if changed or role_changed else "")
    body = f'<div class="now">{e(i.base)}</div>'
    if changed:
        body += f'<div class="was"><b>Was (layered_tuned_v1, in use on the demo):</b> {e(was)}</div>'
    if role_changed:
        body += f'<div class="was"><b>Role before:</b> {e(old_effect(cid) or "")}</div>'
    cls = "rule" + (" changed" if (changed or role_changed or new) else "")
    return f'<li class="{cls}" id="r{n}" data-text="{e((cid + " " + i.base).lower())}"><div class="head"><span class="num">#{n}</span><code>{e(cid)}</code>{label_html}{tag}</div>{body}</li>'


sections = []
asked = prof.asked_criteria()
hidden = set(prof.settings.get("client_rules_hidden") or [])
client_ids = {r["rule_id"] for r in rb.sources["client_rules"]}
for r in rb.sources["client_rules"]:
    rid = r["rule_id"]
    if rid in hidden:
        continue
    items = [rule_row(rid, '<span class="tag client">client\'s own text (shown to the AI for reference)</span>')]
    for c in sorted([c for c in asked if c.parent_rule_id == rid], key=lambda c: (c.priority, c.criterion_id)):
        k, txt = effect(c.criterion_id)
        items.append(rule_row(c.criterion_id, f'<span class="tag {k}">{e(txt)}</span><span class="tag src">{e(SRC.get(c.source, c.source))}</span>'))
    sections.append(f'<section><h2>{e(rid)} · {e(r.get("source_ref", ""))}</h2><ol>{"".join(items)}</ol></section>')
other_parents = {c.parent_rule_id for c in asked} - client_ids - {None}
leftover = [c for c in asked if c.parent_rule_id in hidden or c.parent_rule_id in other_parents or not c.parent_rule_id]
if leftover:       # questions that belong to a client line the profile no longer shows (for example R012, R013, R015) and the proposed extras
    items = []
    for c in sorted(leftover, key=lambda c: (c.parent_rule_id or "zz", c.priority, c.criterion_id)):
        k, txt = effect(c.criterion_id)
        items.append(rule_row(c.criterion_id, f'<span class="tag {k}">{e(txt)}</span><span class="tag src">{e(SRC.get(c.source, c.source))}</span><span class="tag src">{e(c.parent_rule_id or "other")}</span>'))
    sections.append(f'<section><h2>Questions whose client line is no longer shown, and the proposed extra</h2><ol>{"".join(items)}</ol></section>')
facts = sorted([c for c in prof.criteria.values() if c.decision_effect == "FACT" and c.criterion_id in FACT_FIELDS], key=lambda c: c.criterion_id)
sections.append('<section><h2>Scene facts (what the AI reports about the scene first; Python uses them to decide which rules apply)</h2><ol>'
                + "".join(rule_row(c.criterion_id, '<span class="tag info">scene fact</span>') for c in facts) + "</ol></section>")
sections.append(f'<section><h2>Team guidance</h2><p class="lede">None. All four lines were removed ({len(guidance_of(old_prof))} in layered_tuned_v1).</p></section>')

gone = sorted(set(old_cat) - set(cat))
rem = []
for cid in gone:
    why = removals.get(cid) or removals.get(f"{cid} (client line)") or ""
    why = re.sub(r"\s*Not applied\..*$", "", why)
    rem.append(f'<li class="rule"><div class="head"><code>{e(cid)}</code><span class="tag">{e(old_cat[cid].kind.lower().replace("_", " "))}</span></div>'
               f'<div class="was">{e(old_cat[cid].base)}</div><div class="why">{e(why)}</div></li>')
sections.append(f'<section><h2>Removed from layered_tuned_v1 ({len(gone)})</h2><ol>{"".join(rem)}</ol></section>')

hard = sum(1 for g in prof.gates if prof.is_hard(g))
old_hard = sum(1 for g in old_prof.gates if old_prof.is_hard(g))
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Final rules</title>
<style>
:root{{--bg:#fbfbfa;--card:#fff;--ink:#111;--mute:#555;--line:#ddd;--hl:#fff3c4;--red:#8a1c1c;--new:#e3f4e3}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--card:#1e1e1e;--ink:#eee;--mute:#aaa;--line:#333;--hl:#3a3315;--red:#f19a9a;--new:#17301a}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px 80px}} h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:17px;margin:28px 0 8px;border-bottom:2px solid var(--ink);padding-bottom:4px}}
.lede{{color:var(--mute);margin:0 0 14px}} .bar{{position:sticky;top:0;background:var(--bg);padding:10px 0;z-index:2}}
input{{width:100%;box-sizing:border-box;padding:10px 12px;font:inherit;border:1.5px solid var(--ink);background:var(--card);color:var(--ink)}}
ol{{list-style:none;margin:0;padding:0}} .rule{{background:var(--card);border:1px solid var(--line);padding:10px 14px;margin:8px 0}} .rule.changed{{border-left:5px solid var(--ink);background:var(--hl)}}
.head{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:4px}} .num{{font-weight:700;min-width:44px}} code{{font:13px ui-monospace,Consolas,monospace}}
.tag{{font-size:12px;padding:1px 7px;border:1px solid var(--line);color:var(--mute)}} .tag.rejects{{border-color:var(--red);color:var(--red);font-weight:700}} .tag-chg,.tag-new{{font-size:12px;padding:1px 7px;background:var(--ink);color:var(--bg);font-weight:700}}
.was,.why{{font-size:13px;color:var(--mute);margin-top:4px}} .stats{{display:flex;flex-wrap:wrap;gap:14px;font-size:14px;color:var(--mute)}}
</style></head><body><main>
<h1>The final rules (profile {FINAL})</h1><p class="lede">The owner's review of 2026-11-05, compared with {OLD}, the profile in use on the demo today. Nothing here is live yet: the demo still uses {OLD} until you say the sheet feels right. Refer to any rule by its number or id.</p>
<div class="stats"><span><b>{n}</b> rules shown</span><span><b>{len(asked)}</b> questions</span><span><b>{len(facts)}</b> scene facts</span><span><b>{16 - len(hidden)}</b> client lines</span><span><b>0</b> guidance lines</span><span><b>{hard}</b> gates that can reject (was {old_hard}), <b>{len(prof.gates) - hard}</b> soft</span><span><b>{len(gone)}</b> removed</span></div>
<p class="lede">The AI always gives a verdict: a gate it cannot confirm is a rejection ("could not confirm"), quality below 70 is a re-select, a publication blocker is a rejection. A person still decides every photo.</p>
<div class="bar"><input id="q" type="search" placeholder="Filter: type words or an id, e.g. banner, face, R007, #12"></div>
{"".join(sections)}
</main><script>
const q=document.getElementById('q');q.addEventListener('input',()=>{{const t=q.value.trim().toLowerCase();document.querySelectorAll('.rule').forEach(li=>{{const num='#'+li.id.slice(1);li.style.display=!t||(li.dataset.text||li.textContent.toLowerCase()).includes(t)||num===t?'':'none';}});}});
</script></body></html>"""
out = ROOT / "agent-platform" / "reports" / "rules_final.html"
out.write_text(page, encoding="utf-8")
(ROOT / "agent-platform" / "reports" / "rules_final.json").write_text(json.dumps(LIST, ensure_ascii=False, indent=1), encoding="utf-8")
print(out, "| rules shown:", n, "| questions:", len(asked), "| facts:", len(facts), "| removed:", len(gone), "| gates hard/soft:", hard, len(prof.gates) - hard)
