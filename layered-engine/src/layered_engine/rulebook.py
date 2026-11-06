"""Rulebook loading, profile resolution and validation. Everything the engine knows about rules comes from config/rulebook/*.yaml."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .expressions import referenced_names

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
RULEBOOK_DIR = CONFIG_DIR / "rulebook"

SOURCES = ("CLIENT", "TEAM", "PROPOSED")
APPROVAL_STATUSES = ("CLIENT_APPROVED", "TEAM_APPROVED", "PROPOSED", "PENDING_CONFIRMATION", "DEPRECATED")
EFFECTS = ("FACT", "GATE", "DIAGNOSTIC", "SOFT_SCORE", "FLAG", "EXTERNAL", "SUMMARY", "BATCH", "TECHNICAL")


# ------------------------------------------------------------------------------------------------------------------------------------ overlay
# An overlay is a list of change records stored outside the YAML files (the learning loop keeps them in its database). It can change only DATA the engine
# already treats as settings: the quality thresholds and weights, and whether a gate is hard (rejects) or soft (lowers one quality dimension). It can never
# touch the wording of a rule, the questions the model is asked (the prompt is always built from the profile WITHOUT the overlay, so its hash never changes),
# the publication blockers, or which criteria exist. With no overlay everything is exactly as before.
OVERLAY_KINDS = ("QUALITY_THRESHOLDS", "QUALITY_WEIGHTS", "GATE_SOFTEN", "GATE_HARDEN")
MAX_OVERLAY = 50
_DIMENSIONS = ("activity_communication", "technical_quality", "composition", "naturalness_storytelling", "brand_presentation", "visual_polish")


def _number(v: Any, lo: float, hi: float, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise ValueError(f"{what} must be a number from {lo:g} to {hi:g}")
    return float(v)


def validate_overlay(records: Optional[Iterable[Any]]) -> list[dict[str, Any]]:
    """The records, checked for shape (not yet for whether their target exists in a profile: `resolve` checks that). Raises ValueError."""
    out = [dict(r) for r in (records or [])]
    if len(out) > MAX_OVERLAY:
        raise ValueError(f"an overlay holds at most {MAX_OVERLAY} changes")
    seen: set[str] = set()
    for r in out:
        cid, kind, st = r.get("change_id"), r.get("kind"), r.get("set") or {}
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("every overlay change needs a unique change_id")
        seen.add(cid)
        if kind not in OVERLAY_KINDS:
            raise ValueError(f"{cid}: unknown overlay kind {kind!r}; allowed: {list(OVERLAY_KINDS)}")
        if not isinstance(st, dict):
            raise ValueError(f"{cid}: 'set' must be an object")
        if kind == "QUALITY_THRESHOLDS":
            if not st or set(st) - {"review", "reselection"}:
                raise ValueError(f"{cid}: a thresholds change sets 'review' and/or 'reselection'")
            for k, v in st.items():
                _number(v, 0, 100, f"{cid}: {k}")
        elif kind == "QUALITY_WEIGHTS":
            if set(st) != set(_DIMENSIONS):
                raise ValueError(f"{cid}: a weights change sets all of {list(_DIMENSIONS)}")
            if abs(sum(_number(v, 0, 100, f"{cid}: weight") for v in st.values()) - 100) > 1e-9:
                raise ValueError(f"{cid}: the weights must add up to 100")
        else:
            if not isinstance(r.get("target"), str) or not r["target"]:
                raise ValueError(f"{cid}: {kind} needs a target gate id")
            if kind == "GATE_SOFTEN" and st and set(st) - {"cap_max_score"}:
                raise ValueError(f"{cid}: a soften change may only set 'cap_max_score'")
            if "cap_max_score" in st:
                _number(st["cap_max_score"], 0, 4, f"{cid}: cap_max_score")
    return out


class Criterion(BaseModel):
    model_config = ConfigDict(extra="allow")

    criterion_id: str
    source: str
    origin: Optional[str] = None
    parent_rule_id: Optional[str] = None
    gate_id: Optional[str] = None
    description_original: Optional[str] = None
    description_operational: str
    na_condition_text: Optional[str] = None
    layer: int
    decision_effect: str
    priority: int = 100
    applicability: str = "always"
    failure_condition: str = "answer == NO"
    exceptions: list[str] = Field(default_factory=list)
    evidence_required: bool = False
    uncertainty_policy: str = "HUMAN_REVIEW"
    approval_status: str
    version: int = 1
    dimension: Optional[str] = None
    legacy_id: Optional[str] = None
    asked: bool = True
    model_na_allowed: bool = False
    active_in: Optional[list[str]] = None  # None = every profile; a list restricts the criterion to those profiles ([] = none)


class Gate(BaseModel):
    model_config = ConfigDict(extra="allow")

    gate_id: str
    criterion_id: str
    source: str
    parent_rule_id: Optional[str] = None
    instruction: Optional[str] = None
    approval_status: str
    components: list[str] = Field(default_factory=list)
    forced_hard: bool = False  # hard whatever its criterion's own decision_effect says (used when a diagnostic criterion is promoted to a gate)
    active_in: Optional[list[str]] = None
    policy_change: Optional[dict] = None  # recorded when the gate exists only in some profiles


@dataclass
class PolicyChange:
    profile: str
    target: str
    kind: str
    affects_mandatory: bool
    status: str
    reason: str
    change_id: str = ""
    record: dict[str, Any] = field(default_factory=dict)  # original / proposed rule, previous / new effect, evidence (required for the v2 profile)


def _change(profile: str, target: str, pc: dict[str, Any]) -> "PolicyChange":
    return PolicyChange(profile, target, pc["kind"], bool(pc["affects_mandatory"]), pc.get("status", "PROPOSED"), pc["reason"], pc.get("change_id", ""),
                        {k: pc[k] for k in ("original_rule", "proposed_rule", "previous_effect", "new_effect", "evidence") if k in pc})


@dataclass
class ResolvedProfile:
    name: str
    settings: dict[str, Any]
    criteria: dict[str, Criterion]  # effective, enabled criteria only
    gates: list[Gate]  # effective gates (hard or soft is decided by the criterion's decision_effect)
    conditions: dict[str, Any]
    exceptions: list[dict[str, Any]]
    quality: dict[str, Any]
    publication: dict[str, Any]
    taxonomy: list[dict[str, Any]]
    policy_changes: list[PolicyChange] = field(default_factory=list)
    rulebook_version: str = ""
    unknown_is_fail: bool = True
    overlay_ids: list[str] = field(default_factory=list)   # the change_ids of the overlay applied on top of this profile, in order

    @property
    def activity_ids(self) -> list[str]:
        return [a["id"] for a in self.taxonomy]

    def asked_criteria(self) -> list[Criterion]:
        """Criteria the model is asked to answer in Layer 2/3: diagnostics, gates, soft scores."""
        return sorted((c for c in self.criteria.values() if c.asked and c.decision_effect in ("GATE", "DIAGNOSTIC", "SOFT_SCORE")),
                      key=lambda c: (c.layer, c.priority, c.criterion_id))

    def is_hard(self, g: Gate) -> bool:
        return g.forced_hard or self.criteria[g.criterion_id].decision_effect == "GATE"

    def hard_gates(self) -> list[Gate]:
        return [g for g in self.gates if self.is_hard(g)]


class Rulebook:
    def __init__(self, directory: Path = RULEBOOK_DIR) -> None:
        self.directory = Path(directory)
        load = lambda n: yaml.safe_load((self.directory / n).read_text(encoding="utf-8"))  # noqa: E731
        book = load("rulebook_v1.yaml")
        self.version: str = str(book["rulebook_version"])
        self.unknown_is_fail_default: bool = bool(book.get("unknown_is_fail", True))
        self.criteria: dict[str, Criterion] = {c["criterion_id"]: Criterion(**c) for c in book["criteria"]}
        if len(self.criteria) != len(book["criteria"]):
            raise ValueError("duplicate criterion_id in rulebook_v1.yaml")
        self.gates: list[Gate] = [Gate(**g) for g in book["gates"]]
        self.sources = load("rule_sources.yaml")
        self.taxonomy: list[dict[str, Any]] = load("activity_taxonomy.yaml")["activities"]
        self.conditions: dict[str, Any] = load("gate_applicability.yaml")["conditions"]
        self.exceptions: list[dict[str, Any]] = load("exceptions.yaml")["exceptions"]
        self.quality: dict[str, Any] = load("quality_scoring.yaml")
        self.publication: dict[str, Any] = load("publication_checks.yaml")
        self.profiles: dict[str, dict[str, Any]] = load("policy_profiles.yaml")["profiles"]

    # ------------------------------------------------------------------------------------------------ profile resolution
    def resolve(self, profile: str, without: Iterable[str] = (), overlay: Optional[Iterable[Any]] = None) -> ResolvedProfile:
        """`without` names policy changes (their change_id, e.g. "V2-01") to leave out: used for ablation. Everything else is unchanged.
        `overlay`: change records applied on top of the resolved profile (see OVERLAY_KINDS); None or empty leaves the profile exactly as it was."""
        records = validate_overlay(overlay)
        if profile not in self.profiles:
            raise KeyError(f"unknown profile {profile!r}; choose from {sorted(self.profiles)}")
        p = self.profiles[profile]
        lineage = self._lineage(profile)   # the profile and every profile it inherits from: an item active in a parent is active in the child unless the child overrides it
        skip = set(without)
        unknown = skip - self.change_ids(profile)
        if unknown:
            raise KeyError(f"profile {profile!r} has no policy change {sorted(unknown)}; it has {sorted(self.change_ids(profile))}")
        enabled = set(p["enabled_approval_statuses"])
        overrides = self._merged(p, "overrides", skip)
        gate_overrides = self._merged(p, "gate_overrides", skip)
        changes: list[PolicyChange] = []
        criteria: dict[str, Criterion] = {}
        for cid, c in self.criteria.items():
            c = copy.deepcopy(c)
            ov = overrides.get(cid)
            if ov:
                pc = ov.get("policy_change")
                # a PROPOSED override only takes effect in a profile that enables PROPOSED changes (and is not switched off for an ablation)
                if (pc is None or pc.get("status", "PROPOSED") in enabled) and not (pc and pc.get("change_id") in skip):
                    for k, v in ov.items():
                        if k != "policy_change":
                            setattr(c, k, v)
                    if pc:
                        changes.append(_change(profile, cid, pc))
            if c.active_in is not None and not lineage & set(c.active_in):
                continue
            if c.decision_effect == "FACT" and c.layer == 1:
                criteria[cid] = c  # scene facts are infrastructure: Python needs them to compute applicability in every profile
            elif c.approval_status in enabled and c.approval_status != "DEPRECATED":
                criteria[cid] = c
        gates: list[Gate] = []
        for g in self.gates:
            g = copy.deepcopy(g)
            go = gate_overrides.get(g.gate_id)
            if go:
                pc = go.get("policy_change")
                if (pc is None or pc.get("status", "PROPOSED") in enabled) and not (pc and pc.get("change_id") in skip):
                    g.components = list(go.get("components", g.components))
                    g.forced_hard = bool(go.get("forced_hard", g.forced_hard))
                    if pc:
                        changes.append(_change(profile, g.gate_id, pc))
            if g.active_in is not None and not lineage & set(g.active_in):
                continue
            if g.criterion_id in criteria and g.approval_status in enabled:
                gates.append(g)
                if g.policy_change:
                    changes.append(_change(profile, g.gate_id, g.policy_change))
        off = set(p.get("exceptions_off") or [])   # a child profile can switch off an exception its parent has (it cannot do so with active_in, which is inherited)
        exceptions = [e for e in self.exceptions if lineage & set(e.get("active_in", [])) and e["id"] not in off]
        taxonomy = [a for a in self.taxonomy if "active_in" not in a or lineage & set(a["active_in"])]
        settings = copy.deepcopy(p)
        bv = settings.get("beneficiary_visibility")
        if bv and bv.get("change_id") in skip:  # the scope refinement is switched off: back to the legacy subject scope
            bv["subject_scope"] = bv["legacy_subject_scope"]
        quality = self.quality
        if records:
            quality = copy.deepcopy(self.quality)   # the shared configuration is never edited: only this resolved profile sees the overlay
            self._apply_overlay(records, profile, criteria, gates, quality)
        return ResolvedProfile(name=profile, settings=settings, criteria=criteria, gates=gates, conditions=self.conditions, exceptions=exceptions,
                               quality=quality, publication=self.publication, taxonomy=taxonomy, policy_changes=changes,
                               rulebook_version=str(p.get("rulebook_version", self.version)), unknown_is_fail=bool(p.get("unknown_is_fail", True)),
                               overlay_ids=[r["change_id"] for r in records])

    @staticmethod
    def _apply_overlay(records: list[dict[str, Any]], profile: str, criteria: dict[str, Criterion], gates: list[Gate], quality: dict[str, Any]) -> None:
        by_id = {g.gate_id: g for g in gates}
        for r in records:
            cid, kind, st = r["change_id"], r["kind"], r.get("set") or {}
            if kind == "QUALITY_THRESHOLDS":
                th = quality["thresholds"]
                if "review" in st:
                    th["review"] = th["acceptable"] = st["review"]   # the shipped configuration keeps "acceptable" equal to "review"
                if "reselection" in st:
                    th["reselection"] = st["reselection"]
                if not 0 <= th["reselection"] <= th["review"] <= th["strong_candidate"]:
                    raise ValueError(f"{cid}: thresholds must satisfy 0 <= reselection <= review <= {th['strong_candidate']}")
            elif kind == "QUALITY_WEIGHTS":
                for dim, w in st.items():
                    quality["dimensions"][dim]["weight"] = w
            else:
                g = by_id.get(r["target"])
                if g is None:
                    raise ValueError(f"{cid}: profile {profile!r} has no gate {r['target']!r}")
                crit = criteria[g.criterion_id]
                hard = g.forced_hard or crit.decision_effect == "GATE"
                if kind == "GATE_SOFTEN":
                    if not hard:
                        raise ValueError(f"{cid}: gate {g.gate_id} is already soft")
                    crit.decision_effect, g.forced_hard = "SOFT_SCORE", False
                    dim = getattr(crit, "dimension", None)
                    if dim in quality["dimensions"]:   # one defect lowers one dimension once, as for the other soft gates; no dimension means it only informs
                        quality["soft_caps"] = list(quality.get("soft_caps", [])) + [{"criterion_id": g.criterion_id, "dimension": dim, "max_score": st.get("cap_max_score", 2),
                                                                                       "reason": f"softened gate {g.gate_id} ({cid})", "profiles": [profile]}]
                else:   # GATE_HARDEN
                    if hard:
                        raise ValueError(f"{cid}: gate {g.gate_id} is already hard")
                    g.forced_hard = True

    def _lineage(self, profile: str) -> set[str]:
        names, cur = {profile}, self.profiles[profile].get("inherits")
        while cur and cur not in names:
            names.add(cur)
            cur = self.profiles[cur].get("inherits")
        return names

    def _merged(self, p: dict[str, Any], key: str, skip: set[str]) -> dict[str, Any]:
        """The profile's overrides on top of its parent's (`inherits`). A change that is switched off falls back to the parent's entry, if any."""
        # the whole chain of parents, oldest first (a profile that inherits a profile that inherits another keeps the grandparent's overrides too; before layered_final_v1 no
        # profile was more than one step from its root, so this changes nothing for the others). The parent's own entries are taken as they are, as before.
        merged: dict[str, Any] = {}
        if p.get("inherits"):
            parent = self.profiles[p["inherits"]]
            merged = self._merged(parent, key, set()) if parent.get("inherits") else {}
            merged = {**merged, **dict(parent.get(key, {}))}
        for k, ov in p.get(key, {}).items():
            if (ov.get("policy_change") or {}).get("change_id") in skip:
                continue
            merged[k] = ov
        return merged

    def change_ids(self, profile: str) -> set[str]:
        p = self.profiles[profile]
        ids = {ov["policy_change"]["change_id"] for ov in list(p.get("overrides", {}).values()) + list(p.get("gate_overrides", {}).values())
               if ov.get("policy_change", {}).get("change_id")}
        bv = p.get("beneficiary_visibility", {})
        return ids | ({bv["change_id"]} if bv.get("change_id") else set())


# ---------------------------------------------------------------------------------------------------- validation report
@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)
    policy_changes: list[PolicyChange] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def text(self) -> str:
        lines = ["RULEBOOK VALIDATION", ""]
        for k, v in self.counts.items():
            lines.append(f"  {k}: {v}")
        mand = [c for c in self.policy_changes if c.affects_mandatory]
        lines += ["", f"POLICY CHANGES THAT AFFECT A MANDATORY REQUIREMENT ({len(mand)}): these are proposals and need NGO approval"]
        lines += [f"  [{c.profile}] {c.target}: {c.kind} ({c.status}) - {c.reason}" for c in mand]
        other = [c for c in self.policy_changes if not c.affects_mandatory]
        if other:
            lines += ["", f"OTHER POLICY CHANGES ({len(other)})"] + [f"  [{c.profile}] {c.target}: {c.kind} ({c.status}) - {c.reason}" for c in other]
        lines += ["", f"WARNINGS ({len(self.warnings)})"] + [f"  {w}" for w in self.warnings]
        lines += ["", f"ERRORS ({len(self.errors)})"] + [f"  {e}" for e in self.errors]
        lines.append("")
        lines.append("RESULT: " + ("OK" if self.ok else "FAILED"))
        return "\n".join(lines)


LEGACY_GATE_COUNT = 15
LEGACY_QUESTION_COUNT = 65


def validate_rulebook(rb: Rulebook) -> ValidationReport:
    r = ValidationReport()
    crit = rb.criteria
    for c in crit.values():
        if c.source not in SOURCES:
            r.errors.append(f"{c.criterion_id}: unknown source {c.source!r}")
        if c.approval_status not in APPROVAL_STATUSES:
            r.errors.append(f"{c.criterion_id}: unknown approval_status {c.approval_status!r}")
        if c.decision_effect not in EFFECTS:
            r.errors.append(f"{c.criterion_id}: unknown decision_effect {c.decision_effect!r}")
        if c.applicability not in rb.conditions:
            r.errors.append(f"{c.criterion_id}: applicability {c.applicability!r} is not defined in gate_applicability.yaml")
        if c.source == "CLIENT" and c.description_original is None and c.decision_effect != "FACT":
            r.warnings.append(f"{c.criterion_id}: CLIENT source without verbatim client wording")
        if c.source == "PROPOSED" and c.approval_status in ("CLIENT_APPROVED", "TEAM_APPROVED"):
            r.errors.append(f"{c.criterion_id}: a PROPOSED criterion cannot carry an approved status")
        if c.dimension and c.dimension not in rb.quality["dimensions"]:
            r.errors.append(f"{c.criterion_id}: unknown quality dimension {c.dimension!r}")
    for name, node in rb.conditions.items():
        for ref in referenced_names(node):
            if ref not in rb.conditions:
                r.errors.append(f"condition {name}: refers to unknown condition {ref!r}")
    # client wording preserved
    client = {x["rule_id"]: x for x in rb.sources["client_rules"]}
    if len(client) != 16:
        r.errors.append(f"expected 16 client rules, found {len(client)}")
    for c in crit.values():
        if c.source == "CLIENT" and c.description_original and c.parent_rule_id in client:
            if c.description_original != client[c.parent_rule_id]["description_original"]:
                r.errors.append(f"{c.criterion_id}: client wording differs from rule_sources.yaml ({c.parent_rule_id})")
    # legacy inventory
    legacy = [c for c in crit.values() if c.legacy_id]
    if len(legacy) != LEGACY_QUESTION_COUNT:
        r.errors.append(f"expected {LEGACY_QUESTION_COUNT} legacy checklist5 criteria, found {len(legacy)}")
    if "checklist5_legacy" in rb.profiles and len(rb.resolve("checklist5_legacy").hard_gates()) != LEGACY_GATE_COUNT:
        r.errors.append(f"the checklist5_legacy profile must keep exactly {LEGACY_GATE_COUNT} hard gates")
    for g in rb.gates:
        if g.criterion_id not in crit:
            r.errors.append(f"gate {g.gate_id}: unknown criterion {g.criterion_id}")
        for comp in g.components:
            if comp not in crit:
                r.errors.append(f"gate {g.gate_id}: unknown component {comp}")
    # profiles
    for pname, p in rb.profiles.items():
        for cid, ov in p.get("overrides", {}).items():
            if cid not in crit:
                r.errors.append(f"profile {pname}: override for unknown criterion {cid}")
            if ov.get("applicability") and ov["applicability"] not in rb.conditions:
                r.errors.append(f"profile {pname}: override of {cid} uses undefined condition {ov['applicability']!r}")
            if "policy_change" not in ov:
                r.errors.append(f"profile {pname}: override of {cid} is not recorded as a policy_change")
        if p.get("require_change_records"):
            need = ("change_id", "kind", "status", "reason", "original_rule", "proposed_rule", "previous_effect", "new_effect", "evidence")
            for key in ("overrides", "gate_overrides"):
                for tgt, ov in p.get(key, {}).items():
                    pc = ov.get("policy_change") or {}
                    missing = [k for k in need if not pc.get(k)]
                    if missing:
                        r.errors.append(f"profile {pname}: the policy change on {tgt} lacks {missing}")
                    if pc.get("affects_mandatory") and pc.get("status") not in ("PROPOSED_PENDING_NGO_APPROVAL", "TEAM_APPROVED", "CLIENT_APPROVED"):
                        r.errors.append(f"profile {pname}: the policy change on {tgt} that affects a mandatory requirement has status {pc.get('status')!r}")
        for gid in p.get("gate_overrides", {}):
            if gid not in {g.gate_id for g in rb.gates}:
                r.errors.append(f"profile {pname}: gate_override for unknown gate {gid}")
        try:
            res = rb.resolve(pname)
        except Exception as exc:  # noqa: BLE001
            r.errors.append(f"profile {pname}: cannot be resolved: {exc}")
            continue
        r.policy_changes += res.policy_changes
        for e in res.exceptions:
            for need in e.get("requires_criteria_all_yes", []):
                if need not in res.criteria:
                    r.errors.append(f"profile {pname}: exception {e['id']} needs {need}, which is not enabled in this profile")
            for tgt in e["applies_to"]:
                if tgt not in res.criteria:
                    r.errors.append(f"profile {pname}: exception {e['id']} applies to {tgt}, which is not enabled in this profile")
        for g in res.gates:
            for comp in g.components:
                if comp not in res.criteria:
                    r.errors.append(f"profile {pname}: gate {g.gate_id} component {comp} is not enabled")
        r.counts[f"profile {pname}: criteria / hard gates / questions asked"] = f"{len(res.criteria)} / {len(res.hard_gates())} / {len(res.asked_criteria())}"
    by_src: dict[str, int] = {}
    for c in crit.values():
        by_src[c.source] = by_src.get(c.source, 0) + 1
    r.counts["criteria"] = len(crit)
    r.counts["by source"] = by_src
    r.counts["by layer"] = {k: sum(1 for c in crit.values() if c.layer == k) for k in (1, 2, 3, 4)}
    r.counts["legacy checklist5 questions preserved"] = len(legacy)
    r.counts["gates (all profiles)"] = len(rb.gates)
    r.counts["proposed (not yet NGO-approved) criteria"] = sum(1 for c in crit.values() if c.approval_status in ("PROPOSED", "PENDING_CONFIRMATION"))
    return r
