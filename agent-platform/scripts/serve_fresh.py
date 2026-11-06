"""Serve the review interface as a clean first run: its own empty database and photo store, one new organisation, nothing learned.

    python scripts/serve_fresh.py [--port 8100] [--embedder clip|none]

The existing demo (demo_data/demo.db and demo_data/storage) is not opened, copied or changed, so none of its photographs, decisions, tags,
examples or learned changes can influence this instance. The fresh data lives in demo_data/fresh/ (git-ignored) and is KEPT between runs: delete
that folder yourself if you want to start over. Login passwords are random, written once to demo_data/fresh/credentials.txt.

By default evaluation uses the answers the engine already stored for the pilot photographs (found by file hash), so no model is called and no money is
spent; a photograph outside the pilot is refused with a clear message, because answering it would need a real model call.

With --live every uploaded photograph is sent to Gemini (cost about $0.017 each at the placeholder price). It needs the cloud switch and the key in
image-classifier/.env, which are read into memory only and never printed. Start it with --live only when the project owner has said to.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE / "src"), str(HERE.parent / "layered-engine" / "src"), str(Path(__file__).resolve().parent)]

from demo_phase2a import StoredEvaluator  # noqa: E402
from ngo_agent.api.app import create_app  # noqa: E402
from ngo_agent.bootstrap import add_user, create_organisation  # noqa: E402
from ngo_agent.db import create_schema, make_engine, make_session_factory  # noqa: E402
from ngo_agent.embeddings import make_embedder  # noqa: E402
from ngo_agent.gap_service import BaselineGapAnalyzer  # noqa: E402
from ngo_agent.storage import LocalStorage, sha256_hex  # noqa: E402

FRESH = HERE / "demo_data" / "fresh"
ORG = "Fresh NGO (first run)"


class PilotOnlyEvaluator(StoredEvaluator):
    def evaluate(self, image_bytes, mime_type, rulebook_config, metadata, examples=None):  # noqa: ANN001
        if sha256_hex(image_bytes) not in self.by_hash:
            raise RuntimeError("this photograph has no stored answer; evaluating it would need a real model call, which this instance does not make")
        return super().evaluate(image_bytes, mime_type, rulebook_config, metadata, examples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--embedder", default=os.environ.get("DEMO_EMBEDDER", "clip"))
    ap.add_argument("--data-dir", type=Path, default=FRESH, help="where the database, photos and credentials live (use a temporary folder for a throwaway copy)")
    ap.add_argument("--live", action="store_true", help="send uploaded photographs to Gemini (costs money; only when the project owner has said to)")
    ap.add_argument("--llm-gaps", action="store_true", help="after each decision a model reads the reviewer's written reasons (text only, sent to the cloud) instead of the offline keyword matching")
    ap.add_argument("--gap-llm", choices=("gemini", "groq"), default="gemini", help="with --llm-gaps: which model reads them. gemini needs --live; groq needs NGO_ALLOW_GROQ_INFERENCE=true and GROQ_API_KEY in .env")
    ap.add_argument("--gap-max-calls", type=int, default=None, help="with --llm-gaps: stop after this many model calls in this run")
    ap.add_argument("--llm-notes", action="store_true", help="(superseded by --llm-rules, kept for now) with --llm-gaps: make prompt-note drafting available (L5). It still does nothing until an administrator turns notes on in the Learning loop screen; "
                                                             "it is off by default and then drafts only, with nothing applied")
    ap.add_argument("--llm-rules", action="store_true", help="with --llm-gaps: make rule rewriting available (L5). After each decision a model rewrites a rule the reviewer disagreed with the AI about, the client's "
                                                             "own rules included. It does nothing until an administrator turns it on in the Learning loop screen (Rule updates); it is off by default and then drafts only")
    ap.add_argument("--llm-summary", action="store_true", help="with --llm-gaps: a model clubs the AI's findings that say the same thing about the same subject into one sentence on the review screen "
                                                               "(display only; the AI's own findings are sent as text, no photograph and no reviewer text)")
    ap.add_argument("--profile", default="layered_proposed_v1", help="the policy profile the organisation uses (layered_tuned_v1 after the owner's banner changes): the analyst and the rule drafter list that profile's rules")
    ap.add_argument("--vision", choices=("gemini", "haiku"), default=None, help="with --live: which model looks at the photographs. gemini (the default, or the NGO_VISION environment variable) uses gemini_config.yaml; "
                                                                               "haiku uses anthropic_config.yaml (Claude Haiku 5.5: needs ANTHROPIC_API_KEY and NGO_ALLOW_ANTHROPIC_INFERENCE=true in .env and the anthropic package)")
    args = ap.parse_args()
    if args.llm_gaps and args.gap_llm == "gemini" and not args.live:
        ap.error("--llm-gaps with gemini needs --live (it uses the same Gemini authorisation); or use --gap-llm groq")
    if args.llm_notes and not args.llm_gaps:
        ap.error("--llm-notes needs --llm-gaps: notes are drafted from the model-based mapping of the reviewers' reasons")

    gap_analyzer = BaselineGapAnalyzer()
    provider = cfg = None
    if args.live:
        from layered_engine.providers import load_vision_config, make_provider, read_env_file
        from ngo_agent.evaluators import LayeredEngineEvaluator

        cfg = load_vision_config(args.vision)
        env = read_env_file(HERE.parent / "image-classifier" / ".env", (cfg["auth"]["api_key_env"], cfg["auth"]["authorization_env"]))
        provider = make_provider(cfg, env=env)   # raises AuthorizationError if the switch or key is missing: fails closed
        evaluator = LayeredEngineEvaluator(provider, cfg, provider_name="anthropic-claude" if cfg["provider"] == "anthropic" else "google-gemini")
        print(f"LIVE: every uploaded photograph is sent to {'Anthropic (Claude)' if cfg['provider'] == 'anthropic' else 'Gemini'} ({cfg['model_id']}).")
    else:
        evaluator = PilotOnlyEvaluator()
    if args.llm_gaps:
        from layered_engine.providers import read_env_file
        from ngo_agent.groq_text_model import ENV_NAMES
        from ngo_agent.llm_gap_analyzer import make_gap_analyzer

        groq_env = read_env_file(HERE.parent / "image-classifier" / ".env", ENV_NAMES) if args.gap_llm == "groq" else None
        gap_analyzer = make_gap_analyzer("llm", provider=provider, cfg=cfg, max_calls=args.gap_max_calls, llm=args.gap_llm, env=groq_env, profile=args.profile)   # fails closed without its own switch and key
        print(f"LLM GAPS: the reviewer's written reasons (text only, contact details removed) are sent to {'Groq' if args.gap_llm == 'groq' else 'Gemini'} after each decision.")
    note_drafter = None
    if args.llm_notes:
        from layered_engine.rulebook import Rulebook
        from ngo_agent.note_drafter import LlmNoteDrafter

        note_drafter = LlmNoteDrafter(gap_analyzer.model, Rulebook())
        print("PROMPT NOTES: available, but OFF until an administrator turns them on (then drafts only, nothing applied, until APPLY is chosen).")
    rule_drafter = None
    if args.llm_rules:
        if not args.llm_gaps:
            ap.error("--llm-rules needs --llm-gaps: rules are rewritten from the model-based mapping of the reviewers' reasons")
        from layered_engine.rulebook import Rulebook
        from ngo_agent.rule_drafter import LlmRuleDrafter

        rule_drafter = LlmRuleDrafter(gap_analyzer.model, Rulebook(), profile=args.profile)
        print("RULE UPDATES: available, but OFF until an administrator turns them on (then drafts only, nothing applied, until APPLY is chosen).")
    reason_summarizer = None
    if args.llm_summary:
        if not args.llm_gaps:
            ap.error("--llm-summary needs --llm-gaps: it reuses that model")
        from ngo_agent.reason_summary import ReasonSummarizer

        reason_summarizer = ReasonSummarizer(gap_analyzer.model)
        print("REASON SUMMARY: the AI's own findings (text only) are sent to the same model to club the ones that say the same thing; display only.")
    fresh = args.data_dir
    fresh.mkdir(parents=True, exist_ok=True)
    db = fresh / "fresh.db"
    new = not db.exists()
    engine = make_engine(f"sqlite:///{db.as_posix()}")
    create_schema(engine)
    factory = make_session_factory(engine)
    if new:
        pw = {"admin": secrets.token_urlsafe(12), "reviewer": secrets.token_urlsafe(12)}
        with factory() as s:
            org, _ = create_organisation(s, ORG, "admin.fresh@demo.local", pw["admin"])
            add_user(s, org, "reviewer.fresh@demo.local", pw["reviewer"], "REVIEWER")
            s.commit()
        (fresh / "credentials.txt").write_text(f"admin.fresh@demo.local: {pw['admin']}\nreviewer.fresh@demo.local: {pw['reviewer']}\n(local only; created once)\n", encoding="utf-8")
        print("created a new empty database:", db)
    else:
        print("reusing the fresh database (its own history only):", db)
    print("credentials:", fresh / "credentials.txt")
    import uvicorn

    app = create_app(factory, LocalStorage(fresh / "storage"), evaluator, "fresh-secret-" + secrets.token_hex(20), run_inline=False,
                     embedder=make_embedder(args.embedder), gap_analyzer=gap_analyzer, note_drafter=note_drafter, rule_drafter=rule_drafter, reason_summarizer=reason_summarizer)
    print(f"Serving http://127.0.0.1:{args.port} (Ctrl+C to stop)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
