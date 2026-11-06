"""Offline analysis of a checklist run (reads results.jsonl, the pool CSV and the imported human tags; calls nothing).

Three questions, in order:
  1. Do the individual checks differ between approved and disapproved photos (and is the difference real or just which
     programme the photo came from)?
  2. Do the checks, combined by a small model trained on the labels, predict the human decision on held-out photo groups?
  3. For photos a person tagged with a reason, did the matching checks say NO - and how often do they say NO on approved
     photos (a NO that appears everywhere means little)?

Positive class = approved, as everywhere in this project.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .runner import read_results

TAG_TO_CHECKS = {"X1": ["F_PHOTO"], "X2": ["F_SINGLE"]}  # the two non-rule tags the taggers could pick


def load_answers(run_dir, spec: dict[str, Any]) -> pd.DataFrame:
    """One row per photo (the latest successful answer), one column per check holding YES/NO/CANNOT_TELL/NOT_APPLICABLE."""
    latest = {r["image_id"]: r for r in read_results(run_dir) if r.get("ok")}
    ids = [c["id"] for c in spec["checks"]]
    recs = [{"image_id": i, **{c: r["checks"][c]["answer"] for c in ids}} for i, r in latest.items()]
    if not recs:
        return pd.DataFrame(columns=["image_id", *ids]).set_index("image_id")
    return pd.DataFrame(recs).set_index("image_id").sort_index()


def graded_checks(spec: dict[str, Any]) -> list[str]:
    return [c["id"] for c in spec["checks"] if c["rule"] != "FACT"]


def badness(answers: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """Numeric features for a model: 1 where a requirement is NOT met, 0.5 where it could not be told, else 0.

    Only two scene facts are included (a collage or a document is itself informative). The back-view exception, when met,
    clears the back-view checks.
    """
    cols = [c["id"] for c in spec["checks"] if c["rule"] != "FACT"] + ["F_PHOTO", "F_SINGLE"]
    cols = list(dict.fromkeys(cols))
    out = pd.DataFrame(index=answers.index)
    for c in cols:
        out[c] = answers[c].map({"NO": 1.0, "CANNOT_TELL": 0.5}).fillna(0.0)
    # The team's back-view exception overrides the back-view checks: when it is met, a volunteer seen from behind is fine.
    if "T_BACK_EXCEPTION" in answers:
        met = answers["T_BACK_EXCEPTION"] == "YES"
        for c in ("R008_BACK", "T_FRONT"):
            if c in out:
                out.loc[met, c] = 0.0
    return out


def check_table(answers: pd.DataFrame, y: pd.Series, spec: dict[str, Any], programs: pd.Series | None = None) -> pd.DataFrame:
    """Share of photos answering NO to each check, in each class. 'gap' = disapproved minus approved (positive means the
    check fires more on disapproved photos). When programmes are given, 'gap_within_program' averages the gap inside each
    programme that has both classes, so a difference caused only by which programme the photo came from cancels out."""
    rows = []
    rule_of = {c["id"]: c["rule"] for c in spec["checks"]}
    for c in [c["id"] for c in spec["checks"]]:
        a = answers[c]
        graded = a.isin(["YES", "NO"])
        row: dict[str, Any] = {"check": c, "rule": rule_of[c]}
        for name, mask in (("approved", y == 1), ("disapproved", y == 0)):
            g = graded & mask.reindex(a.index, fill_value=False)
            row[f"n_{name}"] = int(g.sum())
            row[f"no_rate_{name}"] = float((a[g] == "NO").mean()) if g.sum() else np.nan
        row["gap"] = row["no_rate_disapproved"] - row["no_rate_approved"]
        if programs is not None:
            gaps, weights = [], []
            for _, idx in programs.groupby(programs).groups.items():
                aa, yy, gg = a.loc[idx], y.loc[idx], graded.loc[idx]
                ma, md = gg & (yy == 1), gg & (yy == 0)
                if ma.sum() >= 3 and md.sum() >= 3:
                    gaps.append((aa[md] == "NO").mean() - (aa[ma] == "NO").mean())
                    weights.append(min(ma.sum(), md.sum()))
            row["gap_within_program"] = float(np.average(gaps, weights=weights)) if gaps else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def cv_auc(X: pd.DataFrame, y: pd.Series, groups: pd.Series, C: float = 0.3, n_splits: int = 5, seed: int = 0) -> dict[str, Any]:
    """Out-of-fold ROC-AUC of an L2 logistic regression on the checks, with whole groups held out together."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    n_groups = groups.nunique()
    if len(y) < 30 or y.nunique() < 2 or n_groups < n_splits:
        return {"auc": np.nan, "note": "too few photos or groups for a cross-validated estimate"}
    X = X.loc[:, X.nunique() > 1]  # a check that never varies carries nothing
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if y.iloc[tr].nunique() < 2:
            continue
        m = LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=seed).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    ok = ~np.isnan(oof)
    return {"auc": float(roc_auc_score(y[ok], oof[ok])), "n": int(ok.sum()), "n_features": int(X.shape[1]),
            "note": "out-of-fold; whole groups held out together"}


def tag_agreement(answers: pd.DataFrame, y: pd.Series, tags: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    """For each human-tagged disapproved photo in the run: which of its tagged rules had a NO among their checks?

    Also returns, per rule, how often the same rule has a NO on APPROVED photos in the run: a rule that says NO about as
    often on approved photos tells us little even when it 'agrees' with the tagger.
    """
    by_rule: dict[str, list[str]] = {}
    for c in spec["checks"]:
        by_rule.setdefault(c["rule"], []).append(c["id"])
    by_rule.update(TAG_TO_CHECKS)

    def any_no(idx, rule) -> bool | None:
        cols = by_rule.get(rule)
        if not cols:
            return None
        return bool((answers.loc[idx, cols] == "NO").any())

    rows = []
    tagged = [i for i in answers.index if i in tags.index and y.get(i) == 0]
    approved = [i for i in answers.index if y.get(i) == 1]
    for i in tagged:
        t = tags.loc[i]
        codes = [c for c in (t["primary"], t["second"], t["third"]) if isinstance(c, str) and c]
        hit = {c: any_no(i, c) for c in codes}
        rows.append({"image_id": i, "primary": codes[0] if codes else None, "human_rules": ",".join(codes),
                     "primary_has_NO": hit.get(codes[0]) if codes else None,
                     "any_tagged_rule_has_NO": any(v for v in hit.values() if v is not None) if hit else None,
                     "n_rules_with_NO": int(sum(any_no(i, r) or 0 for r in by_rule if r != "FACT"))})
    out = pd.DataFrame(rows)
    if len(out):
        base = {r: float(np.mean([any_no(i, r) for i in approved])) if approved else np.nan for r in by_rule if r != "FACT"}
        out.attrs["approved_rule_no_rate"] = base
    return out


def render_report(tbl: pd.DataFrame, cv: dict[str, Any], agree: pd.DataFrame, n: dict[str, int]) -> str:
    f = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"  # noqa: E731
    lines = [f"# Checklist run: {n['photos']} photos ({n['approved']} approved, {n['disapproved']} disapproved)", ""]
    lines += ["## 1. How often each check says NO", "",
              "| check | graded appr/disap | NO rate approved | NO rate disapproved | gap | gap within programme |", "|---|---|---:|---:|---:|---:|"]
    for r in tbl.sort_values("gap", ascending=False).itertuples():
        lines.append(f"| {r.check} | {r.n_approved}/{r.n_disapproved} | {f(r.no_rate_approved)} | {f(r.no_rate_disapproved)} | "
                     f"{f(r.gap)} | {f(getattr(r, 'gap_within_program', np.nan))} |")
    lines += ["", "A positive gap means the check says NO more often on disapproved photos. With so few photos per check,",
              "differences under about 0.15 are noise.", "", "## 2. Do the checks predict the human decision?", "",
              f"Cross-validated AUC: {f(cv.get('auc'))} ({cv.get('note', '')}). 0.5 is chance; the embeddings model reaches about 0.74.", ""]
    if len(agree):
        base = agree.attrs.get("approved_rule_no_rate", {})
        lines += ["## 3. Photos a person tagged with a reason", "",
                  f"{len(agree)} tagged disapproved photos. Primary tagged rule had at least one NO: "
                  f"{agree['primary_has_NO'].dropna().astype(bool).mean():.0%}. Any tagged rule had a NO: "
                  f"{agree['any_tagged_rule_has_NO'].dropna().astype(bool).mean():.0%}.", "",
                  "| primary tag | photos | had a NO on that rule | NO rate on approved photos, same rule |", "|---|---:|---:|---:|"]
        for code, g in agree.groupby("primary"):
            lines.append(f"| {code} | {len(g)} | {g['primary_has_NO'].dropna().astype(bool).mean():.0%} | {f(base.get(code, np.nan))} |")
    return "\n".join(lines)
