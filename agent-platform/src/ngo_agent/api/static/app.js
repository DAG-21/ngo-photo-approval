"use strict";
// Review interface. All data is inserted with textContent (never as HTML markup), so nothing a model or a person wrote can run as markup.
let token = sessionStorage.getItem("token"), me = null;
const $ = (id) => document.getElementById(id);
const app = $("app");

function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") n.className = v; else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) n.append(k.nodeType ? k : document.createTextNode(String(k)));
  return n;
}
const pill = (text, kind) => el("span", { class: "pill " + (kind || "") }, String(text).replaceAll("_", " "));
const kindOf = (s) => ({ YES: "ok", NO: "bad", CANNOT_TELL: "warn", APPROVE: "ok", APPROVED: "ok", APPROVED_PENDING_PUBLICATION_CHECKS: "ok", REJECT: "bad", REJECTED: "bad", HUMAN_REVIEW: "bad", RESELECT: "bad", FAILED: "bad" }[s] || "");
// small line icons, built as DOM nodes (never from markup); sized and stroked in style.css
const ICONS = { check: "M5 12.5l4.5 4.5L19 7", cross: "M6 6l12 12M18 6L6 18", upload: "M12 16V4M7 9l5-5 5 5M4 20h16", image: "M3 4h18v16H3zM3 17l5-4.5 4 3.5 3-2.5 6 5" };
function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"), path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  svg.setAttribute("viewBox", "0 0 24 24"); svg.setAttribute("class", "ico"); svg.setAttribute("aria-hidden", "true"); path.setAttribute("d", ICONS[name]); svg.append(path);
  return svg;
}
function setNav(key) { document.querySelectorAll("#nav button").forEach((b) => b.classList.toggle("on", b.dataset.key === key)); }

async function api(method, path, body, extra) {
  const headers = Object.assign({}, extra || {});
  if (token) headers.Authorization = "Bearer " + token;
  let payload = body;
  if (body && !(body instanceof FormData)) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  const r = await fetch(path, { method, headers, body: payload });
  if (r.status === 401 && path !== "/auth/login") { signOut(); throw new Error("Please sign in again."); }
  const ct = r.headers.get("content-type") || "";
  const data = ct.includes("json") ? await r.json() : await r.blob();
  if (!r.ok) throw new Error((data && data.detail) || r.statusText);
  return data;
}
const imageCache = new Map();
async function imageUrl(photoId) {
  if (!imageCache.has(photoId)) imageCache.set(photoId, URL.createObjectURL(await api("GET", `/photos/${photoId}/image`)));
  return imageCache.get(photoId);
}
function signOut() { stopAsking(); token = null; me = null; sessionStorage.removeItem("token"); render(); }
function notice(box, text, kind) { box.replaceChildren(el("div", { class: "msg " + kind }, text)); }

// ------------------------------------------------------------------------------------------------------------------------ login
function loginView() {
  const err = el("div");
  const email = el("input", { type: "email", autocomplete: "username", required: true }), pw = el("input", { type: "password", autocomplete: "current-password", required: true });
  const form = el("form", { class: "card login", onsubmit: async (e) => {
    e.preventDefault();
    try { const r = await api("POST", "/auth/login", { email: email.value, password: pw.value }); token = r.access_token; sessionStorage.setItem("token", token); render(); }
    catch (x) { notice(err, x.message, "err"); }
  } }, el("h1", {}, "Sign in"), el("label", {}, "Email"), email, el("label", {}, "Password"), pw, el("p", {}, el("button", { type: "submit" }, "Sign in")), err);
  app.replaceChildren(form);
}

// ------------------------------------------------------------------------------------------------------------------------ photo list
const aiApproves = (rec) => rec === "APPROVE" || rec === "APPROVED_PENDING_PUBLICATION_CHECKS";
// The AI always gives a verdict: Approve, or Disapprove. Whatever else the engine may record (re-select for poor quality, or the undecided outcome of an older profile) is shown as Disapprove with its
// reasons; the AI never hands its decision over to a person (a person reviews every photograph anyway).
const aiRejects = (rec) => !!rec && !aiApproves(rec);
const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
// where a photograph stands for the person: waiting for them, decided by them, or not screened
const standing = (p) => ({ APPROVED: "approved", REJECTED: "rejected", FAILED: "failed" }[p.photo_status] || "pending");

function listView(flash) {   // also used directly as a click handler, so only a string counts as a note to show
  setNav("photos");
  const err = el("div"), queue = [], urls = new Map();
  let photos = [], filter = "pending";
  const input = el("input", { type: "file", accept: IMAGE_TYPES.join(","), multiple: true, hidden: true, "aria-label": "Choose photos" });
  const queueBox = el("div"), filterBox = el("div", { class: "filters", role: "group", "aria-label": "Filter photos" }), grid = el("div", { class: "photo-grid" }), emptyBox = el("div"), bulkBox = el("div");

  function addFiles(list) {
    const ok = [...list].filter((f) => IMAGE_TYPES.includes(f.type));
    if (ok.length < list.length) notice(err, "Only JPEG, PNG and WebP photographs can be uploaded; the others were skipped.", "err");
    ok.forEach((f) => { queue.push(f); urls.set(f, URL.createObjectURL(f)); });
    drawQueue();
  }
  function dropQueue() { queue.forEach((f) => URL.revokeObjectURL(urls.get(f))); queue.length = 0; urls.clear(); }
  function drawQueue() {
    if (!queue.length) return queueBox.replaceChildren();
    queueBox.replaceChildren(el("section", { "aria-label": "Ready to upload" },
      el("div", { class: "row" }, el("div", { class: "eyebrow grow" }, `${queue.length} ready to upload`), el("button", { class: "link", onclick: () => { dropQueue(); drawQueue(); } }, "Clear all")),
      el("div", { class: "queue" }, queue.map((f) => el("div", { class: "q" }, el("div", { class: "tile" }, el("img", { src: urls.get(f), alt: "" }),
        el("button", { class: "x", "aria-label": "Remove " + f.name, onclick: () => { queue.splice(queue.indexOf(f), 1); URL.revokeObjectURL(urls.get(f)); urls.delete(f); drawQueue(); } }, icon("cross"))),
        el("div", { class: "name" }, f.name)))),
      el("p", {}, el("button", { onclick: upload }, `Upload and evaluate ${queue.length} photo${queue.length === 1 ? "" : "s"}`))));
  }
  async function upload() {
    const todo = queue.slice(), done = { sent: 0, dup: 0, bad: 0 };
    for (const [i, f] of todo.entries()) {
      notice(err, `Uploading ${i + 1} of ${todo.length}...`, "ok");
      const fd = new FormData(); fd.append("file", f);
      try { const r = await api("POST", "/photos/upload", fd); r.duplicate ? done.dup++ : done.sent++; }
      catch (x) { done.bad++; notice(err, `${f.name}: ${x.message}`, "err"); }
    }
    dropQueue(); drawQueue();
    if (!done.bad) notice(err, `${done.sent} uploaded, the AI evaluation is running.${done.dup ? ` ${done.dup} already uploaded earlier and skipped.` : ""}`, "ok");
    filter = "pending"; load();
  }

  function card(p) {
    const ev = p.latest_evaluation, rec = ev && ev.ai_recommendation, st = standing(p), img = el("img", { alt: "" });
    imageUrl(p.photo_id).then((u) => { img.src = u; }).catch(() => {});
    const badge = !rec ? null : aiApproves(rec) ? el("span", { class: "badge" }, icon("check"), "AI: Approve")
      : el("span", { class: "badge out" }, icon("cross"), "AI: Disapprove");
    const working = ["UPLOADED", "PROCESSING"].includes(p.photo_status);
    const line = st === "approved" ? el("span", {}, icon("check"), " ", el("strong", {}, "Approved")) : st === "rejected" ? el("span", {}, icon("cross"), " ", el("strong", {}, "Rejected"))
      : st === "failed" ? el("strong", {}, "Could not be screened") : el("span", { class: "mute" }, working ? "Screening..." : "Awaiting your review");
    const differs = (st === "approved" || st === "rejected") && rec && aiApproves(rec) !== (st === "approved");
    const open = () => { if (ev && ev.processing_status === "AI_EVALUATED") reviewView(ev.evaluation_id); else load(); };
    const body = [el("span", { class: "tile" }, icon("image"), img, badge),
      el("span", {}, el("span", { class: "pname" }, p.original_filename),
        el("span", { class: "pstatus" }, line, differs ? el("span", { class: "mute small" }, "Differs from AI") : null),
        st === "failed" && ev && ev.error ? el("span", { class: "mute small" }, ev.error) : null)];
    if (st !== "failed") return el("button", { class: "pcard", onclick: open }, ...body);
    // a failed photograph has nothing to open: offer another attempt instead (a new evaluation run; the failed one stays on record)
    const again = el("button", { class: "secondary", onclick: async () => {
      again.disabled = true;
      try { await api("POST", `/photos/${p.photo_id}/evaluations`, {}); notice(err, `${p.original_filename}: the AI evaluation is running again.`, "ok"); filter = "pending"; await load(); }
      catch (x) { again.disabled = false; notice(err, x.message, "err"); }
    } }, "Try again");
    return el("div", { class: "pcard static" }, ...body, again);
  }
  async function retryAll(failed) {
    const cost = (failed.length * 0.017).toFixed(2);   // placeholder price per evaluation, shown so the person knows the order of magnitude before sending
    if (!confirm(`Send ${failed.length} failed photograph${failed.length === 1 ? "" : "s"} to the AI again? Each evaluation costs about $0.017 (about $${cost} in all).`)) return;
    const done = { started: 0, bad: 0 };
    for (const [i, p] of failed.entries()) {
      notice(err, `Restarting ${i + 1} of ${failed.length}...`, "ok");
      try { await api("POST", `/photos/${p.photo_id}/evaluations`, {}); done.started++; } catch (x) { done.bad++; notice(err, `${p.original_filename}: ${x.message}`, "err"); }
    }
    if (!done.bad) notice(err, `${done.started} evaluation${done.started === 1 ? " is" : "s are"} running again.`, "ok");
    filter = "pending"; await load();
  }
  function draw() {
    const n = (k) => photos.filter((p) => k === "all" || standing(p) === k).length;
    const tabs = [["pending", "Pending"], ["approved", "Approved"], ["rejected", "Rejected"], ...(n("failed") ? [["failed", "Failed"]] : []), ["all", "All"]];
    if (!tabs.some(([k]) => k === filter)) filter = "all";
    filterBox.replaceChildren(...tabs.map(([k, label]) => el("button", { "aria-pressed": String(filter === k), onclick: () => { filter = k; draw(); } }, `${label} (${n(k)})`)));
    const shown = photos.filter((p) => filter === "all" || standing(p) === filter);
    bulkBox.replaceChildren(filter === "failed" && shown.length ? el("div", { class: "row bulk" }, el("button", { onclick: () => retryAll(shown) }, `Try all again (${shown.length})`),
      el("span", { class: "mute small" }, "Starts a new AI evaluation for every failed photo below, one after another.")) : "");
    grid.replaceChildren(...shown.map(card));
    emptyBox.replaceChildren(shown.length ? "" : el("div", { class: "empty" }, { pending: "You are all caught up. New photos appear here once the AI has screened them.", approved: "No photos approved yet.", rejected: "No photos rejected yet.", failed: "Nothing failed.", all: "No photos yet. Upload some to get started." }[filter]));
  }
  async function load() {
    ({ photos } = await api("GET", "/photos"));
    draw();
    if (photos.some((p) => ["UPLOADED", "PROCESSING"].includes(p.photo_status))) setTimeout(() => { if (document.body.contains(grid)) load().catch(() => {}); }, 3000);   // keep refreshing while the AI is working
  }

  const zone = el("div", { class: "drop" }, icon("upload"), el("div", { class: "big" }, "Drag photos here"), el("button", { class: "secondary", onclick: () => input.click() }, "Choose photos"),
    el("p", { class: "mute small" }, "Add as many as you like. JPEG, PNG or WebP."), input);
  input.addEventListener("change", () => { addFiles(input.files); input.value = ""; });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("over"); addFiles(e.dataTransfer.files); });

  app.replaceChildren(el("h1", {}, "Photographs"), el("p", { class: "lede" }, "Add photos. The AI screens each one and suggests approve or disapprove, with its reasons. You make the final decision, and nothing is published automatically."),
    zone, queueBox, err, filterBox, bulkBox, grid, emptyBox);
  if (typeof flash === "string") notice(err, flash, "ok");
  load().catch((x) => notice(err, x.message, "err"));
}

// ------------------------------------------------------------------------------------------------------------------------ review
// The reviewer sees only the AI's verdict, its reasons in plain points, and a place to give their own verdict and reasons. The reasons are built from what
// each check found, never from internal ids; the detailed findings stay stored with the evaluation and are not shown here.
// internal ids such as G001_ACTIVITY_VISIBLE never reach the reviewer
const plainText = (s) => String(s || "").replace(/\b[GR]\d{3}(?:_[A-Z0-9]+)+\s*:\s*/g, "").replace(/\(?\b[GR]\d{3}(?:_[A-Z0-9]+)+\b\)?/g, "").replace(/\s+/g, " ").replace(/^[\s;]+|[\s;]+$/g, "");
function verdictPoints(run) {
  const gates = (run.layer_2 || {}).gates || [], text = plainText, points = [];
  gates.filter((g) => g.status === "NO" && g.hard).forEach((g) => points.push(text(g.evidence) || "A mandatory check failed."));
  gates.filter((g) => g.status === "CANNOT_TELL").forEach((g) => points.push("Could not confirm, so it is not approved: " + (text(g.evidence).replace(/\s*-?\s*could not confirm, so the photo is not approved\.?$/i, "") || "a mandatory check was unclear.")));
  if (!aiApproves(run.ai_recommendation)) {   // a low quality score decides without any failed gate: say why, in plain words, from the model's own notes on the soft checks and the quality reason
    gates.filter((g) => g.status === "NO" && !g.hard && g.evidence).forEach((g) => points.push("Photo quality: " + text(g.evidence)));
    const qr = ((run.decision || {}).decision_reasons || []).filter((r) => /^QUALITY:/.test(r));
    if (qr.length) {
      const m = /score ([0-9.]+)/.exec(qr[0]), cut = /(?:cutoff of|threshold) ([0-9.]+)/.exec(qr[0]), dims = (run.layer_3 || {}).dimensions || {};
      const label = { activity_communication: "Showing the activity and the people", technical_quality: "Technical quality (focus, exposure, light)", composition: "Composition and framing",
        naturalness_storytelling: "Naturalness and storytelling", brand_presentation: "Brand presentation (banner, uniform)", visual_polish: "Visual polish and background" };
      const word = ["unusable", "poor", "acceptable", "good", "excellent"];
      const poor = /POOR:([a-z_]+)/.exec(qr[0]), cutNum = cut ? Number(cut[1]) : 70, aboveCutoff = !!m && Number(m[1]) >= cutNum;   // the score can be fine while one area is graded 0 or 1: that disapproves on its own
      points.push(/unusable/i.test(qr[0]) ? "The photo is unusable: a part of its quality is far too low (for example almost black, blown out or too blurred)."
        : aboveCutoff && poor ? `One part of the photo quality is poor: ${label[poor[1]] || poor[1]}. The total, ${Math.round(Number(m[1]))} out of 100, is above the cutoff of ${cutNum}, but a poor part disapproves the photo on its own.`
        : m ? `The overall photo quality is too low: ${Math.round(Number(m[1]))} out of 100, and the cutoff is ${cut ? cut[1] : "70"}.` : "The photo quality is too low.");
      Object.keys(dims).filter((k) => dims[k] && dims[k].applicable !== false && dims[k].score != null && dims[k].score <= 2).sort((a, b) => dims[a].score - dims[b].score).forEach((k) => {
        const why = text(dims[k].explanation);
        points.push(`${label[k] || k}: graded ${dims[k].score} of 4 (${word[dims[k].score]})` + (why ? `. ${why}` : "."));   // the AI's one-sentence reason (the model is asked for one for every grade of 2 or lower)
      });
    }
  }
  if (!points.length && aiApproves(run.ai_recommendation)) points.push("Every mandatory check passed.");
  (run.publication_blockers || []).forEach((b) => points.push("Before publishing: " + String(b).replace(/^[A-Z]+-?\d+\s+/, "")));
  return points.length ? points : ((run.decision || {}).decision_reasons || []);
}

// Every check the AI answered, with its own short reason (a pass too), in one collapsed list under the verdict, so the reviewer can see why for each check without the screen getting crowded.
const CHECK_NAMES = { R001_POSITION: "Banner is fully in the frame", R001_VISIBLE: "Banner is visible and readable", R001_FLAT: "Banner is flat, not crumpled", R002_GIVER_WHITE: "Giver wears a white top",
  R003_FACE: "Faces are visible", R003_FACE_SHARP: "Faces are sharp", R003_CROP: "Beneficiaries are inside the frame", R003_OBSTRUCTION_PPE_ONLY: "A covered face is only protective equipment",
  R004_GENUINE: "Natural, nothing forced (posed is fine)", R006_ITEM_SHARP: "The item being given is sharp", R006_ACTION: "A seva activity is taking place", R006_SIZE: "The main people and the activity are clearly visible",
  R007_ANGLE: "Camera angle is level and at the people's height", R007_CUTOFF: "Nothing important is cut off", R007_DISTORTION: "No wide-angle distortion", R007_PLACEMENT: "Subject is well placed and not blocked",
  R008_BACK: "Nobody is shown fully from behind", R010_TECHNICAL: "Focus, exposure and lighting", R011_BACKGROUND: "Background is not distracting", R012_DIGNITY: "Everyone's dignity is kept",
  R013_HUMANITY: "Humanity and compassion come through", R015_PERSON_PRESENT: "A relevant person is visible", R015_BEN_RECEIVING: "A beneficiary is receiving the item", "NEW-014": "No watermark or overlay" };
function allChecks(run) {
  const answers = ((run.layer_2 || {}).criteria || []).filter((a) => a && a.criterion_id);
  if (!answers.length) return null;
  const mark = { YES: ["✓", "ok"], NO: ["✗", "bad"], CANNOT_TELL: ["?", "warn"], NOT_APPLICABLE: ["–", ""] };
  const rows = answers.map((a) => {
    const m = mark[a.status] || ["", ""], why = plainText(a.evidence);
    return el("li", {}, el("span", { class: "mark " + m[1], "aria-hidden": "true" }, m[0]), " ", el("strong", {}, CHECK_NAMES[a.criterion_id] || a.criterion_id), why ? ": " + why : (a.status === "NOT_APPLICABLE" ? ": not applicable here." : "."));
  });
  return el("details", { class: "allchecks" }, el("summary", {}, `All ${answers.length} checks and the AI's reason for each`), el("ul", { class: "reasonlist" }, rows));
}

// The AI writes one piece of evidence per failed check, so one problem (a toddler whose face is turned away) can come out as four sentences. The server joins the findings that make the same
// point about the same subject (a model does the grouping; nothing is dropped, a lone finding keeps its wording) and the original sentences stay one click away.
async function clubReasons(run, evaluationId, list, box) {
  const gates = (run.layer_2 || {}).gates || [];
  if (!list || !gates.some((g) => (g.status === "NO" && g.hard) || g.status === "CANNOT_TELL")) return;
  let r;
  try { r = await api("GET", `/evaluations/${evaluationId}/summary`); } catch (x) { return; }
  if (!r.available || !r.groups.some((g) => g.ids.length > 1) || !list.isConnected) return;
  const label = (g) => (g.kind === "UNSURE" ? "Could not tell: " : ""), rest = (run.publication_blockers || []).map((b) => "Before publishing: " + String(b).replace(/^[A-Z]+-?\d+\s+/, ""));
  list.replaceChildren(...r.groups.map((g) => el("li", {}, label(g) + g.text)), ...rest.map((x) => el("li", {}, x)));
  box.append(el("details", { class: "small mute" }, el("summary", {}, "Show the AI's original sentences"), el("ul", {}, ...r.groups.flatMap((g) => g.original.map((x) => el("li", {}, label(g) + x))))));
}

async function reviewView(evaluationId) {
  const run = await api("GET", `/evaluations/${evaluationId}`);
  let review = run.my_review;
  const msg = el("div"), imgEl = el("img", { class: "photo", alt: "The photograph under review" }), blockers = run.publication_blockers || [];
  imageUrl(run.photo_id).then((u) => { imgEl.src = u; });
  const rec = run.ai_recommendation, recApproves = aiApproves(rec), recRejects = aiRejects(rec);
  const points = verdictPoints(run);
  const list = points.length ? el("ul", { class: "reasonlist" }, points.map((r) => el("li", {}, r))) : null;
  const verdictSection = el("div", { class: "section" }, el("div", { class: "eyebrow" }, "AI verdict"),
    el("h2", { class: "verdict" }, recApproves || recRejects ? el("span", { class: "dot" + (recApproves ? "" : " out") }, icon(recApproves ? "check" : "cross")) : null,
      recApproves ? "Approve" : "Disapprove"),
    list, allChecks(run));
  clubReasons(run, evaluationId, list, verdictSection);   // when several findings say the same thing, they are joined into one (in the background; the list above is complete without it)

  let decisionSection;
  if (review && review.review_status === "FEEDBACK_SUBMITTED") {
    // a submitted review is final: show it, nothing to edit
    const approved = review.human_final_decision === "APPROVE", ack = (review.events || []).find((e) => e.event_type === "BLOCKER_ACKNOWLEDGED");
    decisionSection = el("div", { class: "section" }, el("div", { class: "eyebrow" }, "Your verdict"), el("h2", { class: "verdict" }, el("span", { class: "dot" + (approved ? "" : " out") }, icon(approved ? "check" : "cross")), approved ? "Approved" : "Disapproved"),
      el("p", { class: "small mute" }, (approved === recApproves && (recApproves || recRejects)) ? "This matches the AI's verdict." : recApproves || recRejects ? "This differs from the AI's verdict." : "Your decision is final."),
      review.reviewer_comments ? el("div", {}, el("div", { class: "eyebrow" }, "Your reasons"), el("p", {}, review.reviewer_comments)) : null,
      ack ? el("p", { class: "small mute" }, "Approved with publication checks outstanding; your reasons were recorded as the explanation.") : null,
      review.low_information_reason ? el("p", { class: "small mute" }, "Your reasons were very short or generic. More specific reasons (what you saw, which rule it affects) help the AI learn more.") : null);
  } else {
    const reason = el("textarea", { id: "reason", placeholder: "Why do you approve or disapprove this photo? Say what matters to you, including anything the AI missed or got wrong." });
    // choosing a verdict does not save anything: Submit stays locked until a verdict is chosen AND the reasons box is long enough, then it saves
    let choice = null;
    const approve = el("button", { class: "secondary", "aria-pressed": "false", onclick: () => pick("APPROVE") }, "Approve the photo"), disapprove = el("button", { class: "secondary", "aria-pressed": "false", onclick: () => pick("REJECT") }, "Disapprove the photo");
    const submit = el("button", { disabled: true, onclick: send }, "Submit"), hint = el("p", { class: "small mute", "aria-live": "polite" });
    const minimum = () => (choice === "APPROVE" && blockers.length ? 15 : 10);   // approving over a publication blocker needs a fuller explanation
    function refresh() {
      const len = reason.value.trim().length, need = minimum();
      approve.setAttribute("aria-pressed", String(choice === "APPROVE")); disapprove.setAttribute("aria-pressed", String(choice === "REJECT"));
      submit.disabled = !(choice && len >= need);
      hint.textContent = !choice ? "Choose Approve or Disapprove, then give your reasons, to unlock Submit." : len < need
        ? `Write at least ${need} characters in the reasons box to unlock Submit${choice === "APPROVE" && blockers.length ? " (this photo still has publication checks outstanding, so say why approving is acceptable)" : ""}.` : "";
    }
    function pick(decision) { choice = decision; refresh(); }
    reason.addEventListener("input", refresh);
    async function send() {
      const text = reason.value.trim(), needsStatement = choice === "APPROVE" && blockers.length > 0;
      if (!choice || text.length < minimum()) return refresh();   // the button is locked in this case; guard anyway
      if (needsStatement && !confirm("Publication checks are still outstanding for this photo:\n\n" + blockers.map((b) => "- " + String(b).replace(/^[A-Z]+-?\d+\s+/, "")).join("\n") + "\n\nApprove it anyway? Your reasons will be recorded as the explanation.")) return;
      approve.disabled = disapprove.disabled = submit.disabled = true;
      try {
        review = review || await api("POST", `/evaluations/${evaluationId}/review`);   // the review is opened only now, so just looking at a photograph records nothing
        await api("POST", `/feedback/${review.review_id}/submit`, { decision: choice, comment: text, publication_blocker_statement: needsStatement ? text : null }, { "Idempotency-Key": review.review_id });
        listView(`Your verdict was saved: the photo is now under ${choice === "APPROVE" ? "Approved" : "Rejected"}.`);   // back to the main page
      } catch (x) { approve.disabled = disapprove.disabled = false; refresh(); notice(msg, x.message, "err"); }
    }
    decisionSection = el("div", { class: "section" }, el("div", { class: "eyebrow" }, "Your verdict"), el("div", { class: "decide" }, approve, disapprove),
      el("label", { for: "reason" }, "Your reasons (required)"), reason, el("p", {}, submit), hint);
    refresh();
  }

  // the similar past decisions the AI was shown for this photograph (most similar first); only present when an administrator has switched examples on
  const shown = (run.few_shot && run.few_shot.examples ? run.few_shot.examples : []).slice().reverse();
  const exampleRow = (e) => {
    const img = el("img", { class: "ex-thumb", alt: "A similar photograph decided earlier" });
    imageUrl(e.photo_id).then((u) => { img.src = u; });
    const row = el("div", { class: "example" }, img, el("div", {}, pill(e.decision === "APPROVE" ? "staff approved" : "staff rejected", kindOf(e.decision)), ` ${Math.round(e.similarity * 100)}% similar`,
      e.tags && e.tags.length ? el("div", { class: "small mute" }, "Tags: " + e.tags.join(", ")) : null, el("div", { class: "small" }, e.reasons || "(no reason recorded)")));
    if (me && me.role === "ORG_ADMIN") row.lastChild.append(el("button", { class: "link", onclick: async () => {
      const why = prompt("Why is this photograph misleading as an example? It will not be shown to the AI again."); if (!why) return;
      try { await api("POST", `/photos/${e.photo_id}/example-flag`, { flagged: true, reason: why }); notice(msg, "Marked misleading: it will not be used as an example again.", "ok"); } catch (x) { notice(msg, x.message, "err"); } } }, "Mark as misleading"));
    return row;
  };
  const examplesCard = shown.length ? el("div", { class: "card" }, el("strong", {}, "Similar past decisions the AI was shown"),
    el("p", { class: "small mute" }, "The AI saw these photographs with the staff's tags and reasons, and was told to judge this photograph on its own. The rules, not the examples, produce the verdict."), shown.map(exampleRow)) : "";

  setNav("photos");
  app.replaceChildren(el("p", {}, el("button", { class: "secondary", onclick: listView }, "Back to photographs")), msg,
    el("div", { class: "grid" }, el("div", { class: "left sticky" }, imgEl, run.integrity_ok === false ? el("div", { class: "msg err" }, "Integrity check failed for this evaluation record.") : null),
      el("div", { class: "right" }, verdictSection, decisionSection, examplesCard)));
}

// ------------------------------------------------------------------------------------------------------------------ "this rule was changed earlier"
// When the loop wants to change a rule that has ALREADY been changed after the rulebook was set, nothing happens until a person answers this card. A rule still in its original
// wording is changed without any card. Everything is inserted as text. The card cannot be dismissed without an answer; the wording in force stays in use until then.
let askTimer = null, asking = false;
function stopAsking() { if (askTimer) clearInterval(askTimer); askTimer = null; asking = false; document.querySelectorAll(".modal-back").forEach((n) => n.remove()); }
async function checkPending() {
  if (asking || !token || !me || document.hidden) return;
  try { const r = await api("GET", `/organisations/${me.organisation_id}/rule-updates/pending`); if (r.pending.length) confirmChange(r.pending[0]); } catch (x) { /* a failed check is simply tried again in a few seconds */ }
}
function startAsking() { stopAsking(); askTimer = setInterval(checkPending, 6000); checkPending(); }

function confirmChange(p) {
  asking = true;
  const before = document.activeElement, err = el("div");
  const answer = async (accept) => {
    keep.disabled = change.disabled = true;
    try { await api("POST", `/organisations/${me.organisation_id}/rule-updates/${p.edit_id}/answer`, { accept }); }
    catch (x) { notice(err, x.message, "err"); keep.disabled = change.disabled = false; return; }
    back.remove(); asking = false; if (before && before.focus) before.focus(); checkPending();
  };
  const keep = el("button", { class: "secondary", onclick: () => answer(false) }, "No, keep the current wording"), change = el("button", { onclick: () => answer(true) }, "Yes, change it again");
  const e = p.earlier_change, when = e && e.when ? e.when.slice(0, 16).replace("T", " ") : "";
  const history = e ? `${p.rule_id}${p.gate_id ? ` (gate ${p.gate_id})` : ""} was changed${when ? " on " + when : " earlier"}${e.by_administrator ? " by " + e.by_administrator : " from reviewers' feedback"}${e.because ? `: ${e.because}` : "."}`
    : `${p.rule_id}${p.gate_id ? ` (gate ${p.gate_id})` : ""} has a change waiting for your answer.`;
  const back = el("div", { class: "modal-back" }, el("div", { class: "modal", role: "dialog", "aria-modal": "true", "aria-labelledby": "modal-title" },
    el("h2", { id: "modal-title" }, "This rule was changed earlier"), el("p", { class: "small mute" }, history),
    el("h4", { class: "section" }, "The rule now"), el("p", {}, p.current),
    el("h4", { class: "section" }, "It would be changed to"), el("p", { class: "now" }, p.proposed),
    p.reviewer_point ? el("p", { class: "small mute" }, `Why: ${p.reviewer_point}${p.photo ? ` (photograph ${p.photo})` : ""}`) : null,
    p.can_reject ? el("p", { class: "small" }, "This rule is behind a gate that can reject a photograph.") : null,
    el("p", { class: "ask" }, "Are you sure you want to change this rule again?"), el("div", { class: "row" }, keep, change), err));
  document.body.append(back);
  keep.focus();
}

async function render() {
  $("logout").hidden = !token;
  if (!token) { $("who").textContent = ""; $("nav").replaceChildren(); return loginView(); }
  try { me = await api("GET", "/me"); } catch (e) { $("nav").replaceChildren(); return loginView(); }
  $("who").textContent = `${me.display_name || me.email} | ${me.role} | ${me.organisation_name}`;
  // the administrator screens (admin.js) are offered to administrators only; the server enforces the role regardless
  $("nav").replaceChildren(el("button", { class: "link", "data-key": "photos", onclick: listView }, "Photographs"), me.role === "ORG_ADMIN" ? el("button", { class: "link", "data-key": "loop", onclick: () => adminView() }, "Learning loop") : "");
  listView();
  startAsking();
}
$("logout").addEventListener("click", signOut);
render();
