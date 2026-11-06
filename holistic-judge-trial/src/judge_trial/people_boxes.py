"""People and activity bounding boxes from Gemini (one call per photo), used to measure brightness around people and how much of the
frame the activity fills. Boxes only: no judgement of the photo, no labels are sent. Resumable JSONL, like the judge runs."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROMPTS_DIR, TrialConfig
from .data import image_path
from .prompt import BuiltPrompt

USER_TEXT = "Return the boxes for this photograph."
ROLES = ("volunteer", "beneficiary", "other")


def boxes_schema() -> dict[str, Any]:
    box = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4}
    person = {"type": "object", "properties": {"role": {"type": "string", "enum": list(ROLES)}, "box": box},
              "required": ["role", "box"], "propertyOrdering": ["role", "box"]}
    return {"type": "object",
            "properties": {"people": {"type": "array", "items": person}, "activity_box": box, "note": {"type": "string"}},
            "required": ["people", "activity_box", "note"], "propertyOrdering": ["people", "activity_box", "note"]}


def build_boxes_prompt(version: str = "boxes1") -> BuiltPrompt:
    system = (PROMPTS_DIR / version / "system.md").read_text(encoding="utf-8").strip()
    schema = boxes_schema()
    digest = hashlib.sha256(json.dumps({"v": version, "s": system, "u": USER_TEXT, "schema": schema}, sort_keys=True).encode()).hexdigest()[:16]
    return BuiltPrompt("B", version, system, "", [], USER_TEXT, schema, [], digest, {}, False, None)


class BoxError(ValueError):
    pass


def _check_box(b: Any) -> list[int]:
    if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 1000 for v in b)):
        raise BoxError(f"bad box {b!r}")
    if b[2] < b[0] or b[3] < b[1]:
        raise BoxError(f"inverted box {b!r}")
    return b


def parse_boxes(text: str) -> dict[str, Any]:
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise BoxError(f"not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("people"), list) or "activity_box" not in raw:
        raise BoxError("missing people/activity_box")
    people = []
    for p in raw["people"]:
        if not isinstance(p, dict) or p.get("role") not in ROLES:
            raise BoxError(f"bad person entry {p!r}")
        people.append({"role": p["role"], "box": _check_box(p.get("box"))})
    return {"people": people, "activity_box": _check_box(raw["activity_box"]), "note": str(raw.get("note", ""))}


def read_rows(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def run(cfg: TrialConfig, ids: list[str], out: Path, backend, prompt: BuiltPrompt, max_calls: int, workers: int = 4) -> dict[str, int]:
    done = {r["image_id"] for r in read_rows(out) if r.get("ok") and r.get("prompt_hash") == prompt.hash}
    todo = [i for i in ids if i not in done]
    if len(todo) > max_calls:
        raise RuntimeError(f"{len(todo)} calls planned but --max-calls is {max_calls}")
    out.parent.mkdir(parents=True, exist_ok=True)
    backend.prepare(prompt)
    ok = failed = 0
    try:
        def one(i):
            row = {"image_id": i, "prompt_hash": prompt.hash}
            try:
                res = backend.judge(image_path(cfg, i).read_bytes(), "image/jpeg")
                row.update(ok=True, usage=res.usage, **parse_boxes(res.text))
            except BoxError as exc:
                row.update(ok=False, error=f"bad response: {exc}")
            except Exception as exc:  # noqa: BLE001
                row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:200]}")
            return row

        with ThreadPoolExecutor(max_workers=workers) as pool, out.open("a", encoding="utf-8") as fh:
            for fut in as_completed([pool.submit(one, i) for i in todo]):
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                ok, failed = ok + bool(row["ok"]), failed + (not row["ok"])
    finally:
        backend.close()
    return {"ok": ok, "failed": failed, "skipped_done": len(done)}


# ------------------------------------------------------------------------------------------------- measurements
def norm_box(b: list[int], h: int, w: int) -> tuple[int, int, int, int]:
    y0, x0, y1, x1 = b
    a, c = int(y0 / 1000 * h), int(x0 / 1000 * w)
    return a, c, max(int(np.ceil(y1 / 1000 * h)), a + 1), max(int(np.ceil(x1 / 1000 * w)), c + 1)


def box_metrics(y: np.ndarray, row: dict, margin: float = 0.3) -> dict[str, float]:
    """Brightness inside people boxes, in the band around them, and in the rest; plus how much of the frame people / the activity fill."""
    h, w = y.shape
    boxes = [norm_box(p["box"], h, w) for p in row["people"]]
    out: dict[str, float] = {"n_people": float(len(boxes))}
    ab = row["activity_box"]
    out["activity_cover"] = ((ab[2] - ab[0]) * (ab[3] - ab[1])) / 1e6 if ab[2] > ab[0] and ab[3] > ab[1] else np.nan
    if not boxes:
        return out
    people = np.zeros((h, w), bool)
    zone = np.zeros((h, w), bool)
    head = np.zeros((h, w), bool)
    for y0, x0, y1, x1 in boxes:
        people[y0:y1, x0:x1] = True
        dy, dx = int((y1 - y0) * margin), int((x1 - x0) * margin)
        zone[max(y0 - dy, 0):min(y1 + dy, h), max(x0 - dx, 0):min(x1 + dx, w)] = True
        head[y0:y0 + max((y1 - y0) // 4, 1), x0:x1] = True
    ring, rest = zone & ~people, ~zone
    g = float(y.mean())
    out.update({"people_cover": float(people.mean()), "tallest_person": max((y1 - y0) / h for y0, _, y1, _ in boxes),
                "median_person_h": float(np.median([(y1 - y0) / h for y0, _, y1, _ in boxes])),
                "y_global": g, "y_people": float(y[people].mean()), "y_people_p10": float(np.percentile(y[people], 10)),
                "y_head": float(y[head].mean()), "y_head_p10": float(np.percentile(y[head], 10)),
                "people_minus_global": float(y[people].mean() - g), "head_minus_global": float(y[head].mean() - g)})
    if ring.sum() > 20:
        out["y_ring"] = float(y[ring].mean())
        out["ring_minus_global"] = out["y_ring"] - g
    if rest.sum() > 20:
        out["y_rest"] = float(y[rest].mean())
        out["people_minus_rest"] = out["y_people"] - out["y_rest"]
        out["ring_minus_rest"] = out.get("y_ring", np.nan) - out["y_rest"]
    return out
