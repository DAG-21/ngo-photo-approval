# NGO Photo Approval: the whole project in one folder

An explainable photo-approval engine for an NGO's seva photographs: a vision model (Gemini) answers each rule about a photograph, **Python** decides, a person decides every photograph,
and the system learns from the reviewers' feedback. Everything below was copied here on 2026-11-05 from the separate folders it used to live in. The originals were not changed or removed.

## What is where

| Folder | What it is |
|---|---|
| `image-classifier/` | The original project: the client's 16 rules (`rules/`), the first classifier, **the Python environment (`.venv/`)**, the **secrets (`.env`)**, the photograph data (`data/`), docs and scripts. The other folders find `.env` and `data/` here. |
| `layered-engine/` | The rule engine: the rulebook (`config/rulebook/*.yaml`), the prompt builder, the decision engine, tests. The policy profiles live in `policy_profiles.yaml` (`layered_proposed_v1` is the frozen original, `layered_tuned_v1` has the owner's banner changes). |
| `agent-platform/` | The web service (FastAPI): review screens, the learning loop, rule rewriting, administrator screens, the demo (`demo_data/`), reports. |
| `holistic-judge-trial/` | The earlier trial of a single "holistic judge" prompt, with its runs and reports. |
| `ngo-photo-approval-engine/` | The earliest Gemini pilot of the pipeline (was `project-3/gemini-try/`). |
| `models/huggingface/hub/` | The two image-embedding models used to find similar photographs (CLIP ViT-B/32 and DINOv2), copied from your user cache so the project does not depend on it. |
| `working-notes/` | The rules-review drafts agreed so far and the scripts that produced the rules sheet. |

The folders sit side by side on purpose: the code finds its neighbours as `../image-classifier` and `../layered-engine`, so the layout must stay as it is.

## Running things (use the project's own environment)

Always call Python as `image-classifier\.venv\Scripts\python.exe -m ...` (the launchers in `.venv\Scripts\` such as `pip.exe` still contain the old folder's path).

* Tests: `.\run_tests.ps1` (all suites), or in one folder: `..\image-classifier\.venv\Scripts\python.exe -m pytest tests -q`
* Demo: `.un_demo.ps1` (offline: no model is called, only the stored pilot answers), `.un_demo.ps1 -Live` (**Claude Haiku 5.5**: photographs go to Anthropic and reviewers' reasons to Groq; costs money) or `.un_demo.ps1 -Live -Vision gemini` (photographs go to Google). **Haiku needs three things that are not in place yet:** `ANTHROPIC_API_KEY` and `NGO_ALLOW_ANTHROPIC_INFERENCE=true` in `image-classifier\.env` (a Claude Pro subscription does not include API calls: the API is billed from the Anthropic Console), and the `anthropic` package in the project's environment (`image-classifier\.venv\Scripts\python.exe -m pip install anthropic`). Without them the start stops and says what is missing; nothing is sent. Log in with the passwords in `agent-platform\demo_dataresh\credentials.txt`.

## Secrets

`image-classifier\.env` (Gemini, Groq and, once added, Anthropic keys and the switches that allow photographs to leave the machine: `NGO_ALLOW_CLOUD_INFERENCE` for Google, `NGO_ALLOW_ANTHROPIC_INFERENCE` for Anthropic, `NGO_ALLOW_GROQ_INFERENCE` for Groq) and `ngo-photo-approval-engine\.env` hold keys. They are in `.gitignore`; never commit or share them.
Copying this folder anywhere copies the keys too.

## What is NOT in this folder (yet)

* **A second physical copy of the photographs.** `image-classifier\data\raw\` (11 GB, 4,793 files) and `image-classifier\data\derived\` (4.3 GB, 7,074 files) did not fit as a second copy in the free disk space (about 10 GB at copy time), so they are **hard links**: every photograph appears as a normal file in this folder, but it shares its bytes with the same file in the original `image-classifier\data\`, so they use the space once. Deleting the original folder later does NOT remove the photographs from this one (the bytes stay until the last name is deleted). The one thing to avoid: editing a photograph in place changes it in both places (nothing in the project does that). To get a fully independent copy later, free about 15 GB and copy `data\raw` and `data\derived` over themselves (copying a hard-linked file onto itself does nothing, so first delete the folder here, then copy from the original).
* **The base Python 3.11.5** (`C:\Python311`). The environment in `.venv` is built on it. Without it the environment does not start.
* **Cloud services:** Gemini (Google), Claude (Anthropic) and Groq, reached over the internet with the keys above; Ollama on this machine is not used by the project.
* Not part of this project, left where they were: `project-1/`, `project-2/`, `grok-sign-lang-app/`, `sign-language-yap&render/`, `Docs for Jio/`.

## State of the work

The owner's rule-by-rule review (all 115 rules) is finished and applied as a new profile, **`layered_final_v1`** (24 questions, 11 gates that can reject, 2 soft gates, 8 scene facts, no guidance lines; the AI always gives a verdict and a person decides every photograph). **The local demo uses it since 2026-11-06** (database backup before the move: `agent-platform/demo_data/fresh/fresh.db.before-final-profile`; the earlier rewrites of the learning loop were withdrawn on purpose). The rules sheet is `agent-platform/reports/rules_final.html` (rules compared with the tuned profile, removals listed). Decisions and removals: `working-notes/agreed_drafts.json`. Design notes and results: `agent-platform/docs/PHASE3_LEARNING_LOOP_DESIGN.md` (section 24). No model has been called with the new profile.
