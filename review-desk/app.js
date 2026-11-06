// Review desk prototype: the photos, each with the AI's verdict and reasons (data/data.js, made by agent-platform/scripts/review_desk_data.py).
// Nothing here is saved or sent anywhere: decisions live in this page only and are gone when it is reloaded.
(function () {
  "use strict";
  const DATA = window.REVIEW_DATA || { photos: [] };
  const photos = DATA.photos;
  const SHOW_FILE_NAMES = true;   // DEMO ONLY: shows each photo's original file name on its card. Set to false (or delete the lines marked FILE NAME) to remove it.
  const REASONS = ["Face not visible", "From behind", "Too dark or blurry", "Wrong angle", "Other"];
  const S = { dec: {}, why: {}, note: {}, filter: "all", activity: "all", sort: "photos", flash: "", flashBad: false };
  const app = document.getElementById("app");

  // ---- small helpers (text always goes in as text, never as markup) ----
  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return e;
  }
  const svg = (inner, size) => { const w = document.createElement("span"); w.innerHTML = `<svg width="${size || 20}" height="${size || 20}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${inner}</svg>`; return w.firstChild; };
  const TICK = "<path d='M5 12l5 5 9-10'/>", CROSS = "<path d='M6 6l12 12M18 6L6 18'/>", LEFT = "<path d='M15 6l-6 6 6 6'/>", RIGHT = "<path d='M9 6l6 6-6 6'/>";
  const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  const plural = (n, w) => `${n.toLocaleString("en-IN")} ${w}${n === 1 ? "" : "s"}`;
  const bad = (p) => p.verdict === "DISAPPROVE";
  const verdictText = (p) => (bad(p) ? "AI: Disapprove" : "AI: Approve");

  const groups = () => {
    const m = new Map();
    for (const p of photos) { if (!m.has(p.activity)) m.set(p.activity, []); m.get(p.activity).push(p); }
    return [...m.entries()].map(([name, list]) => ({ name, slug: slug(name), list, approve: list.filter((p) => !bad(p)).length, disapprove: list.filter(bad).length }));
  };
  const ordered = (list) => list.slice().sort((a, b) => Number(bad(b)) - Number(bad(a)) || a.id.localeCompare(b.id));   // photos the AI disapproved first

  // ---- screen 1: activities ----
  function activitiesView() {
    let gs = groups();
    if (S.activity !== "all") gs = gs.filter((g) => g.slug === S.activity);
    gs.sort(S.sort === "flagged" ? (a, b) => b.disapprove - a.disapprove : (a, b) => b.list.length - a.list.length);
    const all = groups();
    const total = photos.length;
    const sel = (value, opts, on) => h("select", { onchange: (e) => on(e.target.value) }, opts.map(([v, t]) => h("option", { value: v, selected: v === value }, t)));
    return h("main", {},
      h("div", { class: "stack" },
        h("h1", {}, "Photos to review"),
        h("p", { class: "lead" }, `${plural(total, "photo")} in ${plural(all.length, "group")} are waiting for you, grouped by the activity the AI recognised. Videos are not part of this review.`)),
      h("div", { class: "row" },
        h("div", { class: "seg" }, h("button", { class: "on", "aria-pressed": "true" }, `Needs review ${total}`)),
        h("label", { class: "sel" }, "Activity", sel(S.activity, [["all", "All types"], ...all.map((g) => [g.slug, g.name])], (v) => { S.activity = v; render(); })),
        h("label", { class: "sel" }, "Sort", sel(S.sort, [["photos", "Most photos first"], ["flagged", "Most AI: Disapprove first"]], (v) => { S.sort = v; render(); }))),
      h("div", { class: "acts" }, gs.map((g) => {
        const decided = g.list.filter((p) => S.dec[p.id]).length;
        const pct = (n) => (100 * n) / g.list.length + "%";
        return h("a", { class: "act", href: `#/activity/${g.slug}`, "aria-label": `Review ${g.name}, ${plural(g.list.length, "photo")}` },
          h("div", { class: "act-h" },
            h("div", {}, h("div", { class: "act-t" }, g.name), h("div", { class: "act-s" }, "Activity recognised by the AI")),
            h("span", { class: g.disapprove ? "pill bad" : "pill ok" }, g.disapprove ? `${g.disapprove} AI: Disapprove` : "All AI: Approve")),
          h("div", { class: "thumbs" }, ordered(g.list).slice(0, 4).map((p) => h("img", { src: p.file, alt: "" }))),
          h("div", { class: "meter" },
            h("div", { class: "meter-l" }, h("span", { style: "font-weight:600" }, plural(g.list.length, "photo")), h("span", { style: "color:#5A5D65" }, `${decided} decided`)),
            h("div", { class: "bar" }, h("i", { style: `background:#2E8B5C;width:${pct(g.approve)}` }), h("i", { style: `background:#C8554B;width:${pct(g.disapprove)}` })),
            h("div", { class: "legend" }, h("span", {}, `${g.approve} AI: Approve`), h("span", {}, `${g.disapprove} AI: Disapprove`))),
          h("span", { class: "go" }, "Review photos"));
      })),
      h("p", { class: "note" }, `Sample: ${plural(total, "photo")} from the NGO's archive. The AI is ${DATA.model || "Claude Haiku"} with the final rules. Nothing is saved or published from this page.`));
  }

  // ---- screen 2: review an activity ----
  function reviewView(slugName) {
    const g = groups().find((x) => x.slug === slugName);
    if (!g) return h("main", {}, h("a", { class: "back", href: "#/" }, svg(LEFT, 16), "All activities"), h("div", { class: "empty" }, "This group was not found."));
    const list = ordered(g.list);
    const count = (fn) => list.filter(fn).length;
    const decided = count((p) => S.dec[p.id]), approved = count((p) => S.dec[p.id] === "a"), rejected = count((p) => S.dec[p.id] === "r"), undecided = list.length - decided;
    const defs = [["all", `All ${list.length}`], ["bad", `AI: Disapprove ${count((p) => bad(p) && !S.dec[p.id])}`], ["und", `Undecided ${undecided}`], ["a", `Approved ${approved}`], ["r", `Rejected ${rejected}`]];
    const keep = (p) => { const d = S.dec[p.id]; return S.filter === "all" || (S.filter === "bad" && bad(p) && !d) || (S.filter === "und" && !d) || (S.filter === "a" && d === "a") || (S.filter === "r" && d === "r"); };
    const set = (id, v) => { const was = S.dec[id]; if (was === v) { delete S.dec[id]; delete S.why[id]; } else { S.dec[id] = v; if (v !== "r") delete S.why[id]; } S.flash = ""; render(); };
    const cards = list.filter(keep).map((p) => {
      const d = S.dec[p.id];
      return h("div", { class: "card" + (d ? " " + d : "") },
        h("a", { class: "ph", href: `#/photo/${p.id}`, "aria-label": `Open photo ${p.id.slice(1)} full size. ${verdictText(p)}` },
          h("img", { src: p.file, alt: "", loading: "lazy" }), h("span", { class: "tag" + (bad(p) ? " bad" : "") }, verdictText(p))),
        h("div", { class: "cbody" },
          SHOW_FILE_NAMES && p.name && h("div", { class: "fname", title: p.name }, p.name.split("/").pop()),   // FILE NAME (demo)
          h("div", { class: "cnote" + (bad(p) ? " bad" : "") }, p.headline, bad(p) && p.reasons[0] && p.reasons[0].why && h("div", { class: "cwhy" }, p.reasons[0].why + ".")),
          d === "r" && h("div", { class: "why" }, REASONS.map((r) => h("button", { class: S.why[p.id] === r ? "on" : "", onclick: () => { S.why[p.id] = r; S.flash = ""; render(); } }, r))),
          h("div", { class: "dec" },
            h("button", { class: "btn" + (d === "a" ? " a-on" : ""), "aria-label": "Approve photo", "aria-pressed": String(d === "a"), onclick: () => set(p.id, "a") }, svg(TICK), h("span", {}, d === "a" ? "Approved" : "Approve")),
            h("button", { class: "btn rej" + (d === "r" ? " r-on" : ""), "aria-label": "Reject photo", "aria-pressed": String(d === "r"), onclick: () => set(p.id, "r") }, svg(CROSS), h("span", {}, d === "r" ? "Rejected" : "Reject")))));
    });
    const send = () => {
      const need = list.filter((p) => S.dec[p.id] === "r" && !S.why[p.id]).length;
      if (undecided > 0) { S.flash = `${undecided} photos still to decide`; S.flashBad = true; }
      else if (need > 0) { S.flash = `${need} rejected photos need a reason`; S.flashBad = true; }
      else { S.flash = "Sent. This is a prototype: nothing is saved."; S.flashBad = false; }
      render();
    };
    const first = list[0];
    return h("div", {},
      h("main", { class: "review" },
        h("div", { class: "stack", style: "gap:12px" },
          h("a", { class: "back", href: "#/" }, svg(LEFT, 16), "All activities"),
          h("div", { class: "title-row" }, h("h1", {}, g.name), h("span", { class: "sub" }, `${plural(list.length, "photo")} · activity recognised by the AI`)),
          first && first.summary && h("p", { class: "sub", style: "max-width:720px;line-height:1.55;color:#4A4D55" }, `For example: ${first.summary}.`)),
        h("div", { class: "bar-row" },
          h("div", { class: "chips" }, defs.map(([k, t]) => h("button", { class: "chip" + (S.filter === k ? " on" : ""), "aria-pressed": String(S.filter === k), onclick: () => { S.filter = k; render(); } }, t))),
          h("div", { class: "sub" }, "Sorted with AI: Disapprove first.")),
        cards.length ? h("div", { class: "grid" }, cards) : h("div", { class: "empty" }, "No photos match this filter."),
        h("p", { class: "note" }, "Sample photos from the NGO's archive. Decisions here are not saved.")),
      h("div", { class: "sticky" }, h("div", { class: "wrap sticky-in" },
        h("div", { class: "prog" }, h("div", {}, `${decided} of ${list.length} photos decided`), h("div", { class: "prog-b" }, h("i", { style: `width:${(100 * decided) / list.length}%` }))),
        h("div", { class: "row", style: "gap:12px 20px" }, h("span", { class: "flash " + (S.flashBad ? "bad" : "ok"), role: "status" }, S.flash), h("button", { class: "send", onclick: send }, `Send ${approved} approved to Ready to Publish`)))));
  }

  // ---- screen 3: one photo, full size ----
  function photoView(id) {
    const p = photos.find((x) => x.id === id);
    if (!p) return h("main", {}, h("a", { class: "back", href: "#/" }, svg(LEFT, 16), "All activities"), h("div", { class: "empty" }, "This photo was not found."));
    const g = groups().find((x) => x.name === p.activity), list = ordered(g.list), i = list.findIndex((x) => x.id === id);
    const prev = list[i - 1], next = list[i + 1], d = S.dec[id];
    const set = (v) => { if (S.dec[id] === v) { delete S.dec[id]; delete S.why[id]; } else { S.dec[id] = v; if (v !== "r") delete S.why[id]; } render(); };
    const mark = { YES: ["✓", "y"], NO: ["✗", "n"], CANNOT_TELL: ["?", "q"], NOT_APPLICABLE: ["–", ""] };
    const reasons = p.reasons.filter((r) => r.title !== p.headline || !p.lowAreas.length).map((r) => h("li", {}, h("b", {}, r.title), r.why ? `: ${r.why}.` : "."));   // a low-quality headline is explained by the area list below, not repeated
    const low = p.lowAreas.map((a) => h("li", {}, h("b", {}, a.area), `: graded ${a.score} of 4` + (a.why ? `. ${a.why}.` : ".")));
    return h("div", {},
      h("div", { class: "dtop" }, h("div", { class: "dtop-in" },
        h("a", { class: "back", href: `#/activity/${g.slug}` }, svg(LEFT, 16), `Back to ${g.name}`),
        h("div", { class: "pnav" },
          h("a", { href: prev ? `#/photo/${prev.id}` : "#/photo/" + id, "aria-label": "Previous photo", "aria-disabled": String(!prev) }, svg(LEFT, 18)),
          h("span", {}, `Photo ${i + 1} of ${list.length}`),
          h("a", { href: next ? `#/photo/${next.id}` : "#/photo/" + id, "aria-label": "Next photo", "aria-disabled": String(!next) }, svg(RIGHT, 18))))),
      h("div", { class: "det" },
        h("div", { class: "det-l" }, h("div", { class: "big" }, h("img", { src: p.file, alt: "The photograph under review" }))),
        h("aside", { class: "det-r" },
          h("section", { class: "panel" },
            h("span", { class: "verdict" + (bad(p) ? " bad" : "") }, verdictText(p)),
            h("h2", {}, p.headline),
            bad(p) ? h("ul", { class: "rl" }, reasons, low) : h("p", {}, p.summary ? `${p.summary}.` : "Every mandatory check passed."),
            p.quality != null && h("p", { class: "small" }, `Photo quality: ${Math.round(p.quality)} out of 100`),
            h("details", { class: "all" }, h("summary", {}, `All ${p.checks.length} checks and the AI's reason for each`),
              h("ul", { class: "ck" }, p.checks.map((c) => { const m = mark[c.status] || ["", ""]; return h("li", {}, h("span", { class: "m " + m[1], "aria-hidden": "true" }, m[0]), h("b", {}, c.name), c.why ? `: ${c.why}.` : "."); })))),
          h("section", { class: "panel" },
            h("div", { style: "font-size:15px;font-weight:600" }, "Your decision"),
            h("div", { class: "dec" },
              h("button", { class: "btn" + (d === "a" ? " a-on" : ""), "aria-pressed": String(d === "a"), onclick: () => set("a") }, h("span", {}, d === "a" ? "Approved" : "Approve")),
              h("button", { class: "btn rej" + (d === "r" ? " r-on" : ""), "aria-pressed": String(d === "r"), onclick: () => set("r") }, h("span", {}, d === "r" ? "Rejected" : "Reject"))),
            d === "r" && h("div", { style: "display:flex;flex-direction:column;gap:10px" }, h("div", { style: "font-size:14px;color:#4A4D55" }, "Why? Pick at least one reason."),
              h("div", { class: "why", style: "gap:8px" }, REASONS.map((r) => h("button", { style: "padding:8px 14px;font-size:13px", class: S.why[id] === r ? "on" : "", onclick: () => { S.why[id] = r; render(); } }, r)))),
            h("label", {}, "Note for the team (optional)", h("textarea", { rows: "3", placeholder: "For example: the child on the left is turned away", oninput: (e) => { S.note[id] = e.target.value; } }, S.note[id] || "")),
            h("div", { class: "row", style: "justify-content:space-between" },
              h("span", { class: "small" }, d ? "Saved in this prototype only." : "Not decided yet."),
              d && h("button", { style: "border:0;background:transparent;font-size:14px;font-weight:500;text-decoration:underline", onclick: () => set(d) }, "Undo"))))));
  }

  // ---- routing ----
  let lastKey = null;
  function render() {
    const parts = location.hash.replace(/^#\/?/, "").split("/");
    const key = parts.slice(0, 2).join("/");
    const view = parts[0] === "activity" ? reviewView(parts[1]) : parts[0] === "photo" ? photoView(parts[1]) : activitiesView();
    const y = window.scrollY;
    app.replaceChildren(view);
    document.title = parts[0] === "activity" || parts[0] === "photo" ? "Review desk: " + (parts[0] === "photo" ? "photo" : "review an activity") : "Review desk: photos to review";
    if (key !== lastKey) window.scrollTo(0, 0); else window.scrollTo(0, y);
    lastKey = key;
  }
  window.addEventListener("hashchange", render);
  render();
})();
