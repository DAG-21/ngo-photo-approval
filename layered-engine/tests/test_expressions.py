from __future__ import annotations

import pytest

from layered_engine.expressions import ExpressionError, evaluate, referenced_names

C = {"yes": {"fact": "a", "is": "YES"}, "both": {"all": [{"ref": "yes"}, {"fact": "b", "is": "YES"}]}}


def ev(node, facts=None, meta=None):
    return evaluate(node, facts or {}, meta or {}, C)


def test_leaves_and_unknown():
    assert ev({"fact": "a", "is": "YES"}, {"a": "YES"}) is True
    assert ev({"fact": "a", "is": "YES"}, {"a": "NO"}) is False
    assert ev({"fact": "a", "is": "YES"}, {"a": "UNKNOWN"}) is None  # unknown stays unknown
    assert ev({"fact": "a", "is": "YES"}, {}) is None
    assert ev({"fact": "t", "in": ["X", "Y"]}, {"t": "Y"}) is True
    assert ev({"metadata": "ok", "is": True}, {}, {"ok": True}) is True and ev({"metadata": "ok", "is": True}) is None


def test_three_valued_logic():
    assert ev({"all": [{"const": True}, {"fact": "a", "is": "YES"}]}, {"a": "UNKNOWN"}) is None
    assert ev({"all": [{"const": False}, {"fact": "a", "is": "YES"}]}, {"a": "UNKNOWN"}) is False  # a definite false wins
    assert ev({"any": [{"const": True}, {"fact": "a", "is": "YES"}]}, {"a": "UNKNOWN"}) is True
    assert ev({"any": [{"const": False}, {"fact": "a", "is": "YES"}]}, {"a": "UNKNOWN"}) is None
    assert ev({"not": {"fact": "a", "is": "YES"}}, {"a": "UNKNOWN"}) is None and ev({"not": {"const": False}}) is True


def test_refs_and_errors():
    assert ev({"ref": "both"}, {"a": "YES", "b": "YES"}) is True and ev({"ref": "both"}, {"a": "YES", "b": "NO"}) is False
    assert referenced_names({"all": [{"ref": "x"}, {"not": {"ref": "y"}}]}) == {"x", "y"}
    with pytest.raises(ExpressionError):
        ev({"ref": "missing"})
    with pytest.raises(ExpressionError):
        ev({"bogus": 1})
    with pytest.raises(ExpressionError):
        evaluate({"ref": "loop"}, {}, {}, {"loop": {"ref": "loop"}})
