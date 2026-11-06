"use strict";
// Administrator screens: the learning loop (overview and controls), its changelog (including the list held for developers, with rollback) and the
// similar-photo examples switch. Same rules as app.js: everything is inserted as text, never as markup, and the server enforces the administrator role.
const LOOP_FIELDS = [
  ["every_n", "Run a cycle after this many newly decided photographs"],
  ["min_rows", "Fewest usable staff decisions before anything is attempted"],
  ["min_pattern", "Photographs showing the same gap before a gate change is drafted"],
  ["pattern_share", "Share of them where staff went the same way (0.5 to 1)"],
  ["holdout_pct", "Percent of photographs kept out of drafting and used to test"],
  ["min_gain", "Errors a change must remove on the held-out photographs"],
  ["min_rejected_for_loosening", "Staff-rejected held-out photographs needed before a loosening change may be applied"],
  ["monitor_min", "Photographs seen under a new change before it can be rolled back"],
  ["monitor_window", "Photographs after which a change is confirmed"],
  ["error_margin", "More errors than the previous settings over the window that roll a change back"],
];
const CHANGE_KIND = { APPLIED: "ok", MONITORED_OK: "ok", HELD: "warn", PASSED_NOT_APPLIED: "warn", BLOCKED: "bad", ROLLED_BACK: "bad" };
const FILTERS = [["", "All"], ["APPLIED", "Applied"], ["MONITORED_OK", "Confirmed"], ["HELD", "Held for developers"], ["PASSED_NOT_APPLIED", "Waiting for the next cycle"], ["BLOCKED", "Blocked"], ["ROLLED_BACK", "Rolled back"]];
const OUTCOME = { APPLY: "passed every test", HELD: "held for developers", BLOCKED: "blocked" };
const when = (s) => (s || "").slice(0, 16).replace("T", " ");
const num = (x) => (x === null || x === undefined ? "?" : x);
const words = (s) => String(s || "").replaceAll("_", " ").toLowerCase();

async function adminView(tab, flash) {
  tab = tab || "decisions";
  setNav("loop");
  const msg = el("div"), body = el("div");
  const pages = { decisions: ["Decisions", decisionsTab], overview: ["Overview", overviewTab], changes: ["Changes", changesTab], examples: ["Similar-photo examples", examplesTab], rules: ["Rule updates", rulesTab], prompt: ["Current prompt", promptTab] };
  app.replaceChildren(el("h1", {}, "Learning loop"), el("div", { class: "tabs" }, ...Object.entries(pages).map(([k, [n]]) => el("button", { class: k === tab ? "on" : "", onclick: () => adminView(k) }, n))), msg, body);
  if (flash) notice(msg, flash.text, flash.kind || "ok");
  try { await pages[tab][1](body, msg); } catch (x) { notice(msg, x.message, "err"); }
}

// ------------------------------------------------------------------------------------------------------------------------------ overview
function cycleText(c) {
  const s = c.summary || {}, parts = [];
  if (s.rows !== undefined) parts.push(`${s.rows} usable decisions (${s.held_out} held out)`);
  if (s.monitor) parts.push(`watching the last change: ${words(s.monitor)}`);
  (s.candidates || []).forEach((x) => parts.push(`${x.label} (${words(x.kind)}${x.target ? " " + x.target : ""}): ${OUTCOME[x.outcome] || words(x.outcome)}`));
  if (s.applied) parts.push(`applied ${s.applied.label}`);
  if (s.error) parts.push(`error: ${s.error}`);
  return parts.join("; ") || "-";
}

async function overviewTab(body, msg) {
  const base = `/organisations/${me.organisation_id}/learning-loop`, o = await api("GET", base);
  const save = async (changes, text) => { try { await api("PUT", base, changes); adminView("overview", { text }); } catch (x) { notice(msg, x.message, "err"); } };
  const on = o.enabled && !o.paused;
  const ack = el("input", { type: "checkbox" });
  const controls = o.enabled
    ? el("div", { class: "row" }, el("button", { class: "secondary", onclick: () => save({ paused: !o.paused }, o.paused ? "The loop is running again." : "The loop is paused: nothing will change until it is resumed.") }, o.paused ? "Resume" : "Pause"),
        el("button", { onclick: async () => { try { const r = await api("POST", `${base}/run`); adminView("overview", { text: `Cycle result: ${words(r.status)}.` }); } catch (x) { notice(msg, x.message, "err"); } } }, "Run a cycle now"),
        el("button", { class: "danger", onclick: () => save({ enabled: false }, "The loop is off. Changes it already made stay in place; roll them back on the Changes tab.") }, "Turn off"))
    : el("div", {}, el("label", { class: "check" }, ack, "I understand that once it is on, the quality thresholds and which gates reject a photograph will change automatically (each change tested first and rolled back if it does worse), with no approval for each change, and that the NGO has been told."),
        el("p", {}, el("button", { onclick: () => (ack.checked ? save({ enabled: true, confirm_automatic_changes: true }, "The learning loop is on.") : notice(msg, "Tick the box to confirm first.", "err")) }, "Turn on")));
  const inputs = LOOP_FIELDS.map(([k, label]) => [k, label, el("input", { type: "number", step: k === "pattern_share" ? "0.05" : "1", value: o[k] })]);
  const rows = o.recent_cycles.map((c) => el("tr", {}, el("td", {}, c.ordinal), el("td", {}, c.decided_count), el("td", {}, words(c.trigger)),
    el("td", {}, pill(words(c.status), c.status === "APPLIED" ? "ok" : ["ROLLED_BACK", "FAILED"].includes(c.status) ? "bad" : "")), el("td", { class: "small" }, cycleText(c)), el("td", { class: "mute small" }, when(c.started_at))));
  body.replaceChildren(
    el("div", { class: "card" }, el("div", { class: "row" }, pill(o.enabled ? (o.paused ? "paused" : "on") : "off", on ? "ok" : "warn"), el("span", {}, `${o.decided_photographs} decided photographs`),
      el("span", { class: "mute small" }, `Active settings: ${o.active_rulebook_version || "none"}. ` + (o.active_overlay.length ? `Learned changes in force: ${o.active_overlay.join(", ")}.` : "No learned changes in force."))),
      el("p", { class: "small mute" }, "After every few decided photographs the agent drafts a change to the quality thresholds or to whether a gate rejects by itself, tests it on the stored staff decisions, applies at most one that passes, " +
        "and undoes it by itself if the photographs that follow do worse. It never changes the wording of the rules or the questions the model is asked."), controls),
    el("div", { class: "card" }, el("h3", {}, "Settings"), el("div", { class: "settings" }, inputs.flatMap(([, label, input]) => [el("label", {}, label), input])),
      el("p", {}, el("button", { onclick: () => save(Object.fromEntries(inputs.map(([k, , i]) => [k, Number(i.value)])), "Settings saved.") }, "Save settings"))),
    el("div", { class: "card" }, el("h3", {}, "Recent cycles"), o.recent_cycles.length ? el("table", {}, el("thead", {}, el("tr", {}, ...["#", "Decided", "Trigger", "Result", "What happened", "Started"].map((h) => el("th", {}, h)))), el("tbody", {}, rows))
      : el("p", { class: "small mute" }, "No cycle has run yet.")));
}

// ------------------------------------------------------------------------------------------------------------------------------- changes
async function changesTab(body, msg) {
  const base = `/organisations/${me.organisation_id}`;
  let filter = "";
  const list = el("div"), bar = el("div", { class: "row" });
  async function draw() {
    bar.replaceChildren(...FILTERS.map(([k, n]) => el("button", { class: k === filter ? "" : "secondary", onclick: () => { filter = k; draw(); } }, n)));
    const { changes } = await api("GET", `${base}/changelog${filter ? `?status=${filter}` : ""}`);
    list.replaceChildren(...(changes.length ? changes.map(changeCard) : [el("p", { class: "small mute" }, filter ? "No change has this status." : "The loop has not drafted any change yet.")]));
  }
  function changeCard(c) {
    const t = c.latest_test, g = (t && t.guardrails) || {};
    const history = el("ul", { class: "small" }, c.history.map((h) => el("li", {}, `${words(h.status)}: ${h.reason || "-"} (${when(h.at)}${h.by_person ? ", by a person" : ", automatically"})`)));
    const card = el("div", { class: "card change" }, el("div", { class: "row" }, el("strong", {}, c.label), pill(words(c.status), CHANGE_KIND[c.status] || ""), c.loosening ? pill("loosens the rules", "warn") : pill("tightens or keeps", ""),
      el("span", { class: "mute small" }, `${words(c.kind)}${c.target ? " " + c.target : ""}`)), el("p", {}, c.description), el("p", { class: "small mute" }, `Why: ${c.rationale}.`));
    if (t) card.append(el("p", { class: "small" }, `Tested on ${num(t.photographs)} stored staff decisions (${num(t.held_out)} held out). Photographs staff rejected that the system would approve: ${num(t.false_approvals[0])} now, `
      + `${num(t.false_approvals[1])} with this change. Photographs staff approved that it would not approve: ${num(t.false_rejections[0])} now, ${num(t.false_rejections[1])} with this change. `
      + `Errors on the held-out photographs: ${num(t.held_out_errors[0])} now, ${num(t.held_out_errors[1])} with this change.`
      + (g.loosening ? ` It loosens the rules, so it needs ${g.needed_for_loosening} staff-rejected held-out photographs to be applied; there are ${g.held_out_rejected}.` : "")));
    if (c.status === "HELD") card.append(el("p", { class: "small" }, pill("kept for developers", "warn"), " Not applied: the evidence that it is safe is not enough yet. It is tested again at every cycle."));
    if (c.status === "ROLLED_BACK") card.append(el("p", { class: "small" }, pill("rolled back", "bad"), ` ${c.history[c.history.length - 1].reason}. It will not be applied again.`));
    card.append(el("details", {}, el("summary", { class: "small" }, "History"), history));
    if (c.status === "APPLIED" || c.status === "MONITORED_OK") card.append(el("p", {}, el("button", { class: "danger", onclick: async () => {
      const why = prompt("Why are you rolling this change back? The previous settings will be restored and this change will not be applied again."); if (!why) return;
      try { await api("POST", `${base}/changes/${c.change_id}/rollback`, { reason: why }); adminView("changes", { text: `${c.label} was rolled back; the previous settings are active again.` }); } catch (x) { notice(msg, x.message, "err"); }
    } }, "Roll back")));
    return card;
  }
  body.replaceChildren(el("p", { class: "small mute" }, "Every change the loop has drafted, in plain words, with the evidence and the test behind it. \"Held for developers\" are loosening changes it could not yet prove safe; they are never applied by themselves."), bar, list);
  await draw();
}

// ------------------------------------------------------------------------------------------------------------------------------ examples
async function examplesTab(body, msg) {
  const base = `/organisations/${me.organisation_id}/few-shot`, f = await api("GET", base);
  const k = el("input", { type: "number", min: "1", max: "5", step: "1", value: f.k }), ms = el("input", { type: "number", min: "0", max: "1", step: "0.05", value: f.min_similarity });
  const ack = el("input", { type: "checkbox" });
  const save = async (enabled, text) => {
    try { await api("PUT", base, { enabled, k: Number(k.value), min_similarity: Number(ms.value), confirm_cost_increase: enabled && !f.enabled ? ack.checked : false }); adminView("examples", { text }); }
    catch (x) { notice(msg, x.message, "err"); }
  };
  body.replaceChildren(el("div", { class: "card" }, el("div", { class: "row" }, pill(f.enabled ? "on" : "off", f.enabled ? "ok" : "warn"), el("span", { class: "small" }, `Embedder: ${f.embedder || "none configured"}. ${f.photos_embedded} photographs embedded, `
    + `${f.examples_available} past decisions usable as examples, ${f.photos_marked_misleading} marked misleading.`)),
    el("p", { class: "small mute" }, "When on, the most similar past staff decisions of this NGO are shown to the model with every new photograph (balanced between approved and rejected), with their tags and reasons. " +
      "Each evaluation gets three more photographs (measured on 20 pilot photographs: about 46% more input and about 8% more cost), and past photographs and staff reasons are sent to the model."),
    el("label", {}, "How many examples to show (1 to 5)"), k, el("label", {}, "Lowest similarity allowed (0 to 1; 0 means no floor)"), ms,
    f.enabled ? el("p", { class: "row" }, el("button", { onclick: () => save(true, "Saved.") }, "Save"), el("button", { class: "danger", onclick: () => save(false, "Similar-photo examples are off.") }, "Turn off"))
      : el("div", {}, el("label", { class: "check" }, ack, "I understand each evaluation will carry three more photographs (about 46% more input, about 8% more cost), and that past photographs and staff reasons will be sent to the model."),
        el("p", {}, el("button", { onclick: () => (ack.checked ? save(true, "Similar-photo examples are on.") : notice(msg, "Tick the box to confirm first.", "err")) }, "Turn on")))));
}


// -------------------------------------------------------------------------------------------------------------------------------- rule updates
const RULE_KIND = { APPLIED: "ok", CONFIRMED: "ok", DRAFTED: "warn", HELD: "warn", NEEDS_CONFIRM: "warn", REJECTED: "bad", WITHDRAWN: "bad", NO_CHANGE: "", SUPERSEDED: "", DECLINED: "" };
const RULE_TARGET = { RULE_QUESTION: "question", SCENE_FACT: "scene fact", CLIENT_RULE: "client rule text", GUIDANCE: "team guidance" };
const RULE_MODES = [["OFF", "Off: nothing is rewritten"], ["DRAFT_ONLY", "Drafts only: rewrites are written and shown here, nothing changes in the prompt"], ["APPLY", "Apply: a rewrite that passes every check replaces the rule's wording at once"]];
const RULE_FIELDS = [["max_edits", "Most rewritten rules at once", "1"], ["max_chars", "Longest rewritten wording (characters)", "10"], ["watch_window", "Decisions under a rewrite before it is judged", "1"],
  ["watch_min_before", "Decisions before a rewrite needed to compare against", "1"], ["watch_margin", "Withdraw a rewrite when the share of overridden decisions rises by this much (0.25 = 25 points)", "0.05"]];

async function rulesTab(body, msg) {
  const base = `/organisations/${me.organisation_id}/rule-updates`, o = await api("GET", base), st = o.settings;
  const save = async (changes, text) => { try { await api("PUT", base, changes); adminView("rules", { text }); } catch (x) { notice(msg, x.message, "err"); } };
  const mode = el("select", {}, ...RULE_MODES.map(([k, label]) => el("option", { value: k, selected: k === st.mode }, label)));
  const ack = el("input", { type: "checkbox" });
  const fields = RULE_FIELDS.map(([k, label, step]) => [k, label, el("input", { type: "number", step, value: st[k] })]);
  const byId = Object.fromEntries(o.rules.map((r) => [r.rule_id, r]));
  const pick = el("input", { type: "text", list: "rule-ids", placeholder: "Start typing a rule id, for example R001_FLAT", autocomplete: "off" });
  const ids = el("datalist", { id: "rule-ids" }, ...o.rules.map((r) => el("option", { value: r.rule_id }, `${RULE_TARGET[r.kind] || r.kind}${r.gate_id ? ", gate " + r.gate_id : ""}`)));
  const current = el("p", { class: "small mute" }, "Choose a rule to see its wording.");
  const wording = el("textarea", { rows: "4" }), why = el("input", { type: "text" });
  const dir = el("select", {}, el("option", { value: "NEUTRAL" }, "Only makes the meaning clearer"), el("option", { value: "LOOSEN" }, "Makes the rule pass more often"), el("option", { value: "TIGHTEN" }, "Makes the rule pass less often"));
  pick.addEventListener("input", () => {
    const r = byId[pick.value.trim()];
    current.textContent = r ? `Wording now: ${r.now}` + (r.now !== r.original ? `  |  Original: ${r.original}` : "") : "Choose a rule to see its wording.";
    if (r && !wording.value.trim()) wording.value = r.now;
  });
  const applyBtn = el("button", { onclick: async () => {
    try { await api("POST", `${base}/manual`, { rule_id: pick.value.trim(), wording: wording.value, reason: why.value, direction: dir.value }); adminView("rules", { text: `${pick.value.trim()} was changed.` }); } catch (x) { notice(msg, x.message, "err"); }
  } }, "Change the rule");
  const withdraw = (d) => el("button", { class: "danger", onclick: async () => {
    const why = prompt("Why are you taking this rewrite out? The rule goes back to its original wording."); if (!why) return;
    try { await api("POST", `${base}/${d.edit_id}/withdraw`, { reason: why }); adminView("rules", { text: `The rewrite of ${d.target_id} was withdrawn.` }); } catch (x) { notice(msg, x.message, "err"); }
  } }, "Withdraw");
  const card = (d) => el("div", { class: "card change" }, el("div", { class: "row" }, el("strong", {}, d.target_id), pill(RULE_TARGET[d.target_kind] || words(d.target_kind), ""), d.gate_id ? pill(`gate ${d.gate_id}${d.hard_gate ? "" : " (soft)"}`, "") : null,
    pill(words(d.status), RULE_KIND[d.status] || ""), d.administrator ? pill("set by an administrator", "") : null, ["LOOSEN", "TIGHTEN"].includes(d.direction) ? pill(d.direction === "LOOSEN" ? "loosens" : "tightens", d.direction === "LOOSEN" ? "warn" : "") : null,
    d.flags.includes("FLAGGED_FOR_REVIEW") ? pill("flagged for review", "bad") : null, el("span", { class: "mute small" }, `${words(d.outcome)} | ${when(d.created_at)}`)),
    d.wording ? el("div", {}, el("p", { class: "small mute" }, `Before: ${d.previous}`), el("p", {}, `After: ${d.wording}`)) : null,
    d.flags.length ? el("p", { class: "small" }, (["APPLIED", "CONFIRMED", "SUPERSEDED", "WITHDRAWN"].includes(d.status) ? "Flagged because: " : "Would be flagged because: ") + [d.flags.includes("LOOSENS_GATE") ? "it loosens a gate that can reject a photograph" : "", d.flags.includes("CLIENT_RULE") ? "it changes a rule from the client's own document" : ""].filter(Boolean).join(" and ")) : null,
    d.rationale ? el("p", { class: "small mute" }, `Why: ${d.rationale}`) : null,
    el("p", { class: "small mute" }, `From the reviewer's reasons: ${(d.source && d.source.summary) || "-"}${d.source && d.source.confidence ? ` (mapping confidence ${String(d.source.confidence).toLowerCase()})` : ""}`),
    (d.checks && d.checks.failed || []).length ? el("p", { class: "small" }, `Checks failed: ${d.checks.failed.join("; ")}`) : null,
    el("details", {}, el("summary", { class: "small" }, "History"), el("ul", { class: "small" }, d.history.map((h) => el("li", {}, `${words(h.status)}: ${h.reason || "-"} (${when(h.at)}${h.by_person ? ", by a person" : ", automatically"})`)))),
    ["APPLIED", "CONFIRMED"].includes(d.status) ? el("p", {}, withdraw(d)) : null);
  body.replaceChildren(
    el("div", { class: "card" }, el("div", { class: "row" }, pill(st.paused ? "paused" : words(st.mode), st.mode === "OFF" || st.paused ? "warn" : "ok"), el("span", { class: "small" }, `Prompt in use: ${o.prompt_version || "none"}. ${o.active_rewrites.length} rule(s) carry a rewritten wording.`)),
      el("p", { class: "small mute" }, "After each decision a model reads the reviewer's reasons for a rule the AI got wrong and rewrites that rule's wording. Any rule can change, the client's own rules included. A rewrite that loosens a gate that can " +
        "reject a photograph, or that changes a rule from the client's document, is applied but flagged. Each applied rewrite is watched and taken out by itself if reviewers override the AI more often afterwards. The original wording is always kept: " +
        "withdrawing a rewrite puts it back."),
      o.drafter_available ? null : el("p", { class: "small" }, pill("not available", "bad"), " No model is set up to rewrite rules on this server (start it with --llm-rules, or set AGENT_RULE_DRAFTER=llm)."),
      el("label", {}, "Mode"), mode,
      el("label", { class: "check" }, ack, "I understand the reviewers' written reasons are sent to a cloud model as text, and that in Apply mode the AI's rules, the client's own included, change automatically after every decision, with no approval for each change; the NGO has been told."),
      el("div", { class: "row" }, el("button", { onclick: () => save({ mode: mode.value, confirm_ngo_informed: ack.checked }, `Mode: ${words(mode.value)}.`) }, "Save mode"),
        st.mode !== "OFF" ? el("button", { class: "secondary", onclick: () => save({ paused: !st.paused }, st.paused ? "Rule updates are running again." : "Rule updates are paused: nothing is rewritten or judged until they are resumed.") }, st.paused ? "Resume" : "Pause") : null)),
    el("div", { class: "card" }, el("h3", {}, "Change a rule yourself"),
      el("p", { class: "small mute" }, "Your wording is used from the next photograph, whatever the mode above. The automatic loop will not replace it and the automatic watch will not withdraw it; you can withdraw it below. " +
        "The same safety checks apply: one paragraph, no other rule's id, no instructions to the AI, and a question stays a question."),
      el("label", {}, "Rule"), pick, ids, current, el("label", {}, "New wording"), wording, el("label", {}, "Why are you changing it?"), why, el("label", {}, "Effect on the rule"), dir,
      el("p", {}, applyBtn)),
    el("div", { class: "card" }, el("h3", {}, "Limits"), el("div", { class: "settings" }, fields.flatMap(([, label, input]) => [el("label", {}, label), input])),
      el("p", {}, el("button", { onclick: () => save(Object.fromEntries(fields.map(([k, , i]) => [k, Number(i.value)])), "Limits saved.") }, "Save limits"))),
    el("h3", {}, `Rewritten rules in use now (${o.active_rewrites.length})`), ...(o.active_rewrites.length ? o.active_rewrites.map(card) : [el("p", { class: "small mute" }, "No rule has been rewritten: the AI reads every rule as originally written.")]),
    el("h3", {}, "Every rewrite"), ...(o.drafts.length ? o.drafts.map(card) : [el("p", { class: "small mute" }, "Nothing has been drafted yet.")]));
}


// -------------------------------------------------------------------------------------------------------------------------------- the current prompt
async function promptTab(body, msg) {
  const o = await api("GET", `/organisations/${me.organisation_id}/prompt`);
  const changed = o.rewrites.map((r) => r.now).filter(Boolean);
  const pre = el("pre", { class: "prompt-text" }, ...o.system.split("\n").map((ln) => (changed.some((c) => ln.includes(c)) ? el("span", { class: "hl" }, ln + "\n") : ln + "\n")));
  const copy = el("button", { class: "secondary", onclick: async () => {
    try { await navigator.clipboard.writeText(o.system + "\n\n" + o.user); notice(msg, "The prompt was copied.", "ok"); } catch (x) { notice(msg, "Could not copy automatically: select the text below and copy it.", "err"); }
  } }, "Copy the prompt");
  const rewrites = o.rewrites.map((r) => el("div", { class: "card change" }, el("div", { class: "row" }, el("strong", {}, r.rule_id), r.gate_id ? pill(`gate ${r.gate_id}`, "") : null),
    el("p", { class: "small mute" }, `Original: ${r.original}`), el("p", {}, `Now: ${r.now}`)));
  body.replaceChildren(
    el("div", { class: "card" }, el("div", { class: "row" }, pill(`prompt in use`, "ok"), el("span", { class: "small" }, "Fingerprint "), el("code", {}, o.hash),
      pill(o.matches_registered ? "matches the saved version" : "differs from the saved version", o.matches_registered ? "" : "bad")),
      el("p", { class: "small" }, `${o.rules_asked} questions are put to the AI for every photograph. ${o.rewrites.length} rule(s) currently carry a rewritten wording${o.rewrites.length ? ", highlighted below" : ""}. Profile: ${o.profile}. ${o.characters.toLocaleString()} characters.`),
      el("p", { class: "small mute" }, "This is the fixed text the AI reads for every photograph, built from the rulebook plus any rewritten rules. The photograph itself is added separately, and so are similar past decisions when the examples switch is on and any are close enough. " +
        "Rule rewrites show up here at once."),
      el("p", {}, copy)),
    o.rewrites.length ? el("div", {}, el("h3", {}, "Rules changed from the rulebook's wording"), ...rewrites) : null,
    el("h3", {}, "The prompt"), pre,
    el("h3", {}, "The short message sent with every photograph"), el("pre", { class: "prompt-text" }, o.user));
}


// -------------------------------------------------------------------------------------------------------------------------------- decisions
const CHANGE_STATE = { CHANGED: ["Changed", "ok"], CHANGED_THEN_REPLACED: ["Changed, then replaced by a newer wording", "ok"], CHANGED_THEN_WITHDRAWN: ["Changed, then withdrawn", "bad"],
  REJECTED: ["Rewrite rejected by the checks", "bad"], NOT_NEEDED: ["No change needed", ""], HELD: ["Held for developers", "warn"], WAITING_FOR_CONFIRMATION: ["Waiting for someone to confirm changing it again", "warn"], DECLINED: ["Not changed: someone chose to keep the current wording", ""], DRAFTED_ONLY: ["Drafted, not applied (drafts-only mode)", "warn"], NOT_CHANGED: ["Not changed", ""] };

function aiSide(ai) {
  const lines = ai.points.length
    ? ai.points.map((p) => el("li", {}, el("strong", {}, p.status === "NO" ? "Failed: " : "Could not tell: "), p.question || p.rule_id, p.evidence ? ` (${p.evidence})` : ""))
    : (ai.reasons.length ? ai.reasons.map((r) => el("li", {}, r)) : [el("li", {}, "Every mandatory check passed.")]);
  return el("div", { class: "side" }, el("h4", {}, "The AI"), pill(ai.verdict || "no verdict", kindOf(ai.verdict)), el("ul", {}, ...lines));
}

function humanSide(h) {
  return el("div", { class: "side" }, el("h4", {}, "The reviewer"), pill(h.verdict, kindOf(h.verdict)),
    el("span", { class: "small mute" }, h.agrees_with_ai ? " agreed with the AI" : " disagreed with the AI"), el("p", { class: "reason" }, h.reason || "(no reason written)"));
}

function changeBox(r) {
  const c = r.change, [label, kind] = CHANGE_STATE[c.state] || [words(c.state), ""];
  const live = ["CHANGED", "CHANGED_THEN_REPLACED", "CHANGED_THEN_WITHDRAWN"].includes(c.state);
  const wordLine = !c.after ? r.wording : (c.before !== r.wording ? `Original wording: ${r.wording}` : null);   // the original is only repeated when the wording in force differed from it
  const detail = !c.after ? [c.why ? el("p", { class: "small mute" }, c.why) : null]
    : live ? [el("p", { class: "small was" }, `Was: ${c.before}`), el("p", { class: "now" }, (c.state === "CHANGED" ? "Now: " : "Was changed to: ") + c.after)]
    : [el("p", { class: "small" }, `Still in force: ${c.before}`), el("p", { class: "small was" }, `Proposed, not used: ${c.after}`)];
  return el("div", { class: "rulebox" },
    el("div", { class: "row" }, el("span", { class: "small mute" }, "Relates to rule "), el("code", {}, r.rule_id), r.gate_id ? pill(`gate ${r.gate_id}${r.can_reject ? "" : " (soft)"}`, "") : null),
    wordLine ? el("p", { class: "small mute" }, wordLine) : null,
    el("div", { class: "row" }, pill(label, kind), live && c.flags.length ? pill("flagged for review", "bad") : null), ...detail,
    c.failed_checks.length ? el("p", { class: "small" }, `Checks failed: ${c.failed_checks.join("; ")}`) : null);
}

function disagreementPart(it) {
  if (it.human.agrees_with_ai) return el("p", { class: "small mute" }, "The reviewer agreed with the AI, so there is nothing to change.");
  if (!it.disagreements.length) return el("p", { class: "small mute" }, "The reviewer disagreed with the AI, but no specific rule was identified. " + it.analysis.summary);
  return el("div", {}, ...it.disagreements.map((d) => el("div", { class: "point" },
    el("p", {}, el("strong", {}, "Disagreed on: "), d.point), d.how_it_differs ? el("p", { class: "small mute" }, d.how_it_differs) : null,
    ...(d.rules.length ? d.rules.map(changeBox) : [el("p", { class: "small mute" }, "It could not be tied to a specific rule.")]))));
}

async function decisionsTab(body, msg) {
  const o = await api("GET", `/organisations/${me.organisation_id}/loop-activity?limit=50`);
  const card = (it) => {
    const img = el("img", { class: "loop-thumb", alt: "The photograph that was decided" });
    imageUrl(it.photo_id).then((u) => { img.src = u; }).catch(() => {});
    return el("div", { class: "card loop" },
      el("div", {}, img, el("p", { class: "small mute" }, it.filename), el("p", { class: "small mute" }, when(it.decided_at))),
      el("div", {}, el("div", { class: "sides" }, aiSide(it.ai), humanSide(it.human)),
        el("h4", { class: "section" }, "Where they disagreed, and what changed"), disagreementPart(it),
        it.other_changes.length ? el("div", {}, el("h4", { class: "section" }, "Other rule changes from this decision"), ...it.other_changes.map((x) => changeBox({ ...x, kind: "", gate_id: null, can_reject: true }))) : null,
        el("p", { class: "small mute" }, it.analysis.model ? `Reasons analysed by ${it.analysis.model}: ${it.analysis.summary}` : it.analysis.summary)));
  };
  const ru = o.rule_updates;
  body.replaceChildren(
    el("p", { class: "small mute" }, `Every decided photograph, newest first: what the AI said, what the reviewer decided, where they disagreed, the rule that point relates to and what it was changed to. Rule updates are ${ru.paused ? "paused" : words(ru.mode)}. ` +
      (o.total > o.items.length ? `Showing the latest ${o.items.length} of ${o.total}.` : `${o.total} decision(s).`)),
    ...(o.items.length ? o.items.map(card) : [el("p", { class: "small mute" }, "No photograph has been decided yet.")]));
}
