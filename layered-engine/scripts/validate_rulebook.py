"""Validate the rulebook and print the validation report (offline).

    python scripts/validate_rulebook.py [--write docs/VALIDATION_REPORT.md]

Fails (exit 1) on any error: unknown sources/statuses, undefined conditions, changed client wording, a missing legacy ID, an override that is not
recorded as a policy change. Policy changes that touch a mandatory requirement are listed so they can be taken to the NGO.
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from layered_engine.rulebook import Rulebook, validate_rulebook


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", default=None)
    args = ap.parse_args()
    report = validate_rulebook(Rulebook())
    text = report.text()
    print(text)
    if args.write:
        Path(args.write).write_text(text + "\n", encoding="utf-8")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
