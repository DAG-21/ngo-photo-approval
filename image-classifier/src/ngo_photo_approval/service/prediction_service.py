"""Interface-independent prediction service (used by the CLI, batch runner and FastAPI app).

Pipeline: image validation -> preprocessing -> visual rule analysis -> structured validation ->
rule policy -> probability estimation -> threshold decision -> audit log.

Failure contract: the service NEVER returns APPROVE when something went wrong. Invalid images,
provider failures, malformed model output and missing model artifacts raise typed exceptions
(``InvalidImageError``, ``ProviderError``, ``ResponseValidationError``, ``ModelArtifactError``,
``AuthorizationError``); batch mode records them as error rows instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ModelConfig, PolicyConfig, ProjectConfig
from ..exceptions import (
    AuthorizationError,
    InvalidImageError,
    ModelArtifactError,
    ModelUnavailableError,
    NgoPhotoError,
    ProviderAuthError,
    ProviderError,
    ResponseValidationError,
    TransientProviderError,
)
from ..models.baseline_analyzer import AnalysisResult, RuleAnalyzer
from ..rules.evaluator import evaluate_rules
from ..rules.loader import config_warnings
from ..rules.policy import decide
from ..schemas import (
    CheckOut,
    Decision,
    PolicyResult,
    PredictionResponse,
    ProbabilityType,
    ReasonCode,
    RuleCheck,
    RuleSet,
)
from ..scoring.feature_builder import (
    build_features,
    compute_quality,
    feature_names,
    heuristic_score,
)
from ..scoring.predict import ProbabilityEstimate, ProbabilityScorer

logger = logging.getLogger(__name__)


def anonymous_image_id(path: Path) -> str:
    """ID derived from the file *name* only, so no directory structure leaks into outputs."""
    return "img_" + hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:12]


@dataclass
class Assessment:
    policy_result: PolicyResult
    estimate: ProbabilityEstimate | None
    decision: Decision
    reason_codes: list[ReasonCode]
    heuristic: float | None


def assess(
    rule_set: RuleSet,
    policy: PolicyConfig,
    checks: Sequence[RuleCheck],
    program: str | None,
    scorer: ProbabilityScorer | None,
    feature_set: str,
    quality: Mapping[str, float] | None,
    threshold: float | None = None,
) -> Assessment:
    """Rule policy -> features -> probability -> decision.

    The single code path shared by live inference and offline evaluation, which guarantees that
    evaluated behaviour equals deployed behaviour.
    """
    policy_result = evaluate_rules(rule_set, checks, policy, program)
    estimate: ProbabilityEstimate | None = None
    if scorer is not None:
        feats = build_features(rule_set, feature_set, policy_result, quality)
        estimate = scorer.score(feats)
    prob = estimate.probability if estimate else None
    ptype = estimate.probability_type if estimate else ProbabilityType.NONE
    decision, codes = decide(policy_result, prob, ptype, policy, threshold)
    return Assessment(policy_result, estimate, decision, codes,
                      heuristic_score(policy_result, len(rule_set.active(program))))


class PredictionService:
    def __init__(
        self,
        rule_set: RuleSet,
        policy: PolicyConfig,
        project: ProjectConfig,
        model_cfg: ModelConfig,
        analyzer: RuleAnalyzer | None,
        scorer: ProbabilityScorer | None,
        feature_set: str = "rules",
        model_version: str | None = None,
    ) -> None:
        self.rule_set, self.model_cfg, self.project = rule_set, model_cfg, project
        # REVIEW is possible only if BOTH the project switch and the policy allow it.
        self.policy = policy.model_copy(update={
            "allow_review_status": bool(policy.allow_review_status and project.review.get("enabled", False))})
        self.analyzer, self.scorer, self.feature_set = analyzer, scorer, feature_set
        self.model_version = model_version
        if scorer is not None:
            scorer.check_features(feature_names(rule_set, feature_set))

    # ------------------------------------------------------------------ readiness
    def readiness(self) -> dict[str, Any]:
        return {
            "rule_set_loaded": True,
            "rule_set_version": self.rule_set.rule_set_version,
            "rule_set_confirmed": self.rule_set.rule_set_confirmed,
            "analyzer_available": self.analyzer is not None,
            "probability_model_loaded": self.scorer is not None,
            "calibration_status": self.scorer.metadata.get("calibration_status") if self.scorer else None,
            "warnings": config_warnings(self.rule_set),
        }

    # ------------------------------------------------------------------ single image
    def predict_path(self, image_path: Path, image_id: str | None = None, program: str | None = None) -> PredictionResponse:
        image_path = Path(image_path)
        image_id = image_id or anonymous_image_id(image_path)
        if self.analyzer is None:
            raise AuthorizationError(
                "No visual analyzer is available (cloud inference not authorized or not configured).")

        result = self.analyzer.analyze(image_path, image_id, program)
        self._raise_if_failed(result)
        assert result.analysis is not None

        quality = compute_quality(image_path) if (self.scorer is not None and "quality" in self.feature_set) else None
        a = assess(self.rule_set, self.policy, result.analysis.checks, program, self.scorer,
                   self.feature_set, quality)
        policy_result, estimate, decision, codes, hs = a.policy_result, a.estimate, a.decision, a.reason_codes, a.heuristic
        prob = estimate.probability if estimate else None
        ptype = estimate.probability_type if estimate else ProbabilityType.NONE

        resp = PredictionResponse(
            image_id=image_id,
            decision=decision,
            approval_probability=prob,
            probability_type=ptype,
            threshold=self.policy.approval_threshold,
            hard_rule_failures=policy_result.hard_rule_failures,
            reason_codes=codes,
            checks=[CheckOut(rule_id=c.rule_id, status=c.status, evidence=c.evidence) for c in policy_result.checks],
            rule_set_version=self.rule_set.rule_set_version,
            rule_set_confirmed=self.rule_set.rule_set_confirmed,
            model_id=result.model_id,
            model_version=self.model_version,
            scoring_model_version=estimate.scoring_model_version if estimate else None,
            raw_classifier_probability=estimate.raw_classifier_probability if estimate else None,
            heuristic_score=None if hs is None else round(hs * 100.0, 2),
            self_reported_confidence_uncalibrated=result.analysis.self_reported_confidence,
            warnings=self._warnings(ptype),
        )
        logger.info("decision image=%s decision=%s p=%s type=%s codes=%s", image_id, decision.value,
                    None if prob is None else round(prob, 4), ptype.value, [c.value for c in codes])
        return resp

    def predict_bytes(self, data: bytes, suffix: str = ".jpg", image_id: str | None = None,
                      program: str | None = None) -> PredictionResponse:
        """For uploads: written to a private temp file that is always removed."""
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="ngo_upload_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            return self.predict_path(Path(tmp), image_id=image_id or "upload_" + hashlib.sha1(data).hexdigest()[:12],
                                     program=program)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                logger.warning("could not remove temporary upload file")

    # ------------------------------------------------------------------ batch
    def predict_batch(
        self,
        items: Iterable[tuple[str, Path, str | None]],
        out_jsonl: Path,
        resume: bool = True,
    ) -> dict[str, int]:
        """Process (image_id, path, program) items, appending one JSON record per image.

        Resumable: images with a successful record are skipped; images whose last record is an
        error are retried. Results are flushed per image so a crash loses nothing.
        """
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        done = {r["image_id"] for r in load_batch_results(out_jsonl) if r["status"] == "ok"} if resume else set()
        stats = {"processed": 0, "skipped_done": 0, "ok": 0, "errors": 0}
        if out_jsonl.exists() and out_jsonl.stat().st_size:
            with open(out_jsonl, "rb") as probe:
                probe.seek(-1, os.SEEK_END)
                torn = probe.read(1) != b"\n"
            if torn:  # a crash left a partial last line: terminate it so new records are not glued onto it
                with open(out_jsonl, "a", encoding="utf-8") as fix:
                    fix.write("\n")
        with open(out_jsonl, "a", encoding="utf-8") as fh:
            for image_id, path, program in items:
                if image_id in done:
                    stats["skipped_done"] += 1
                    continue
                stats["processed"] += 1
                try:
                    resp = self.predict_path(path, image_id, program)
                    rec: dict[str, Any] = {"status": "ok", **json.loads(resp.model_dump_json())}
                    stats["ok"] += 1
                except AuthorizationError:
                    raise  # configuration problem: stop the whole batch
                except NgoPhotoError as exc:
                    rec = {"status": "error", "image_id": image_id, "error_type": type(exc).__name__,
                           "error": str(exc)[:300], "retryable": isinstance(exc, TransientProviderError)}
                    stats["errors"] += 1
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
        return stats

    # ------------------------------------------------------------------ helpers
    def _warnings(self, ptype: ProbabilityType) -> list[str]:
        w = []
        if not self.rule_set.rule_set_confirmed:
            w.append("rule set is an unconfirmed draft; decisions are not client-validated")
        if ptype == ProbabilityType.UNCALIBRATED_CLASSIFIER:
            w.append("approval probability is an uncalibrated classifier output")
        if ptype == ProbabilityType.NONE:
            w.append("no probability model is loaded; threshold step cannot be applied")
        return w

    @staticmethod
    def _raise_if_failed(result: AnalysisResult) -> None:
        if result.ok:
            return
        msg = result.error or "analysis failed"
        if result.error_type == "invalid_image":
            raise InvalidImageError(msg)
        if result.error_type == "invalid_response":
            raise ResponseValidationError(msg)
        cls = {"ProviderAuthError": ProviderAuthError, "ModelUnavailableError": ModelUnavailableError,
               "TransientProviderError": TransientProviderError}.get(result.error_class or "", ProviderError)
        raise cls(msg)


def load_batch_results(path: Path) -> list[dict[str, Any]]:
    """Last record per image_id (later records supersede earlier retries)."""
    if not Path(path).exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn final line after a crash
        latest[rec["image_id"]] = rec
    return list(latest.values())


def items_from_directory(directory: Path, extensions: Iterable[str]) -> list[tuple[str, Path, str | None]]:
    exts = {e.lower() for e in extensions}
    return [(anonymous_image_id(p), p, None) for p in sorted(Path(directory).rglob("*"))
            if p.is_file() and p.suffix.lower() in exts]


def build_service(
    project: ProjectConfig, model_cfg: ModelConfig, policy: PolicyConfig, rule_set: RuleSet,
    scoring_model: str = "default", use_tuned: bool = False, feature_set: str = "rules",
    backend: Any | None = None, require_scorer: bool = False,
) -> PredictionService:
    """Factory used by the CLI/API. Builds a Gemini backend only if cloud inference is authorized."""
    from ..models.cloud_client import GeminiBackend
    from ..models.tuned_model import resolve_tuned_model_id

    scorer = None
    art = project.artifacts_dir / "probability_model" / scoring_model
    try:
        scorer = ProbabilityScorer.load(art)
        feature_set = scorer.metadata.get("feature_set", feature_set)
    except ModelArtifactError:
        if require_scorer:
            raise
        logger.warning("probability model not found; probabilities will be unavailable")

    analyzer = None
    if backend is not None or project.cloud.allow_cloud_inference:
        backend = backend or GeminiBackend(project.cloud)
        model_id = resolve_tuned_model_id(model_cfg) if use_tuned else model_cfg.models["primary"].id
        analyzer = RuleAnalyzer(backend, rule_set, model_cfg, model_id)
        version = model_cfg.tuned_model.version if use_tuned else None
    else:
        version = None
    return PredictionService(rule_set, policy, project, model_cfg, analyzer, scorer, feature_set, version)
