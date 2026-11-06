"""Three-valued applicability expressions (true / false / unknown).

Unknown (None) means "the facts needed to decide are UNKNOWN". The engine treats unknown as applicable, so an unclear scene
can never switch a mandatory gate off.

Leaves : {const: bool} {fact: key, is: V} {fact: key, in: [V..]} {metadata: key, is: V} {ref: condition_name}
Combine: {all: [..]} {any: [..]} {not: node}
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

Tri3 = Optional[bool]


class ExpressionError(ValueError):
    pass


def _norm(v: Any) -> Any:
    return v.value if hasattr(v, "value") else v


def evaluate(node: Mapping[str, Any], facts: Mapping[str, Any], metadata: Mapping[str, Any], conditions: Mapping[str, Any], _depth: int = 0) -> Tri3:
    if _depth > 20:
        raise ExpressionError("expression nesting too deep (cycle in refs?)")
    if not isinstance(node, Mapping) or len(node) < 1:
        raise ExpressionError(f"bad expression node: {node!r}")
    if "const" in node:
        return bool(node["const"])
    if "ref" in node:
        name = node["ref"]
        if name not in conditions:
            raise ExpressionError(f"unknown condition {name!r}")
        return evaluate(conditions[name], facts, metadata, conditions, _depth + 1)
    if "fact" in node:
        value = _norm(facts.get(node["fact"], "UNKNOWN"))
        if value in (None, "UNKNOWN"):
            return None
        if "is" in node:
            return value == node["is"]
        if "in" in node:
            return value in node["in"]
        raise ExpressionError(f"fact leaf needs 'is' or 'in': {node!r}")
    if "metadata" in node:
        key = node["metadata"]
        if key not in metadata or metadata[key] is None:
            return None
        return _norm(metadata[key]) == node["is"]
    if "not" in node:
        v = evaluate(node["not"], facts, metadata, conditions, _depth + 1)
        return None if v is None else (not v)
    if "all" in node:
        vals = [evaluate(n, facts, metadata, conditions, _depth + 1) for n in node["all"]]
        if any(v is False for v in vals):
            return False
        return None if any(v is None for v in vals) else True
    if "any" in node:
        vals = [evaluate(n, facts, metadata, conditions, _depth + 1) for n in node["any"]]
        if any(v is True for v in vals):
            return True
        return None if any(v is None for v in vals) else False
    raise ExpressionError(f"unknown expression node: {node!r}")


def referenced_names(node: Mapping[str, Any]) -> set[str]:
    """Every condition name an expression refers to (used by the rulebook validator)."""
    out: set[str] = set()
    if "ref" in node:
        out.add(node["ref"])
    for key in ("all", "any"):
        for n in node.get(key, []):
            out |= referenced_names(n)
    if "not" in node:
        out |= referenced_names(node["not"])
    return out
