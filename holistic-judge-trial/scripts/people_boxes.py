"""Ask Gemini ONLY for bounding boxes of the people and the activity (dry run unless --execute).

    python scripts/people_boxes.py plan                               # builds ids/boxes_sample.csv and prints the plan; no API call
    python scripts/people_boxes.py run --execute --max-calls 160      # real calls (set NGO_AUTH_MODE=api_key for the Gemini API key)

Sample: every disapproved photo tagged R009 (any slot) in train+validation, plus about 100 approved photos matched by programme.
No labels are sent; the model only sees the photo and a fixed instruction.
"""

import argparse
import sys

import _bootstrap  # noqa: F401
import pandas as pd

from judge_trial import data
from judge_trial.config import TRIAL_ROOT, load_trial_config
from judge_trial.people_boxes import build_boxes_prompt, run

SAMPLE = TRIAL_ROOT / "ids" / "boxes_sample.csv"
OUT = TRIAL_ROOT / "analysis" / "people_boxes.jsonl"


def plan(cfg, n_appr=100, seed=5):
    _, _, unique = data.load_main(cfg)
    u = unique[unique.split.isin(["train", "validation"])].set_index("image_id")
    tags = pd.read_csv(cfg.main_root / "data" / "annotations" / "disapproval_reasons.csv")
    r9 = tags[tags[["primary", "second", "third"]].eq("R009").any(axis=1) & tags.image_id.isin(u.index)].image_id.tolist()
    g = pd.DataFrame({"image_id": r9, "group": "R009", "program": [u.loc[i, "program"] for i in r9]})
    appr = u[u.label == "approved"].reset_index()
    picks = []
    for prog, k in g.program.value_counts().items():
        pool = appr[appr.program == prog]
        n = min(len(pool), max(1, int(round(k * n_appr / len(g)))))
        picks.append(pool.sample(n=n, random_state=seed)[["image_id", "program"]].assign(group="appr"))
    return pd.concat([g, *picks], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run"])
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--max-calls", type=int, default=160)
    args = ap.parse_args()
    cfg = load_trial_config()
    prompt = build_boxes_prompt()
    if args.cmd == "plan" or not SAMPLE.exists():
        plan(cfg).to_csv(SAMPLE, index=False, encoding="utf-8")
    s = pd.read_csv(SAMPLE)
    print(f"sample: {len(s)} photos ({(s.group == 'R009').sum()} tagged R009, {(s.group == 'appr').sum()} approved); prompt hash {prompt.hash}")
    print(pd.crosstab(s.program, s.group).to_string())
    if args.cmd == "plan" or not args.execute:
        print("MODE: DRY RUN - no API call is made")
        return 0
    from judge_trial.backend import GeminiJudgeBackend

    backend = GeminiJudgeBackend(cfg)  # fails closed without the main project's authorization
    summary = run(cfg, s.image_id.tolist(), OUT, backend, prompt, args.max_calls)
    print(summary, "->", OUT)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
