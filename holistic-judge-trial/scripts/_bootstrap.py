"""Makes ``import judge_trial`` work when a script is run directly (no install needed)."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The client's rules contain curly apostrophes; the default Windows console encoding cannot print them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
