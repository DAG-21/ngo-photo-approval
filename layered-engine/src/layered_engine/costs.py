"""What one model call cost, from the usage the provider reports and the configured price list.

Billing facts this follows: the model's thinking is billed as output (it is reported separately from the answer, so it must be added), and input tokens served from a cache
are billed at the lower cached rate when one is configured (otherwise at the full input rate, so the estimate never comes out too low). Prices are per model
(`cost.per_model`); a configuration without that table falls back to the flat `input_per_million_usd` / `output_per_million_usd` it always had.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def prices(cfg: Mapping[str, Any], model_id: Optional[str] = None) -> Optional[dict[str, Optional[float]]]:
    c = cfg.get("cost") or {}
    p = (c.get("per_model") or {}).get(model_id or cfg.get("model_id"))
    if p:
        return {"input": p.get("input"), "output": p.get("output"), "cached_input": p.get("cached_input")}
    if c.get("input_per_million_usd") is None or c.get("output_per_million_usd") is None:
        return None
    return {"input": c["input_per_million_usd"], "output": c["output_per_million_usd"], "cached_input": None}


def cost_usd(usage: Mapping[str, Any], cfg: Mapping[str, Any], model_id: Optional[str] = None) -> Optional[float]:
    p = prices(cfg, model_id)
    if p is None or p["input"] is None or p["output"] is None or usage.get("prompt_tokens") is None or usage.get("output_tokens") is None:
        return None
    cached = min(int(usage.get("cached_tokens") or 0), int(usage["prompt_tokens"]))
    written = min(int(usage.get("cache_write_tokens") or 0), int(usage["prompt_tokens"]) - cached)   # Anthropic: input written to the cache is billed above the plain input rate
    fresh = int(usage["prompt_tokens"]) - cached - written
    out = int(usage["output_tokens"]) + int(usage.get("thinking_tokens") or 0)
    cached_rate = p["cached_input"] if p["cached_input"] is not None else p["input"]
    return round((fresh * p["input"] + cached * cached_rate + written * write_rate(cfg, model_id) + out * p["output"]) / 1e6, 8)   # 8 places: a cheap call costs a few hundredths of a cent


def write_rate(cfg: Mapping[str, Any], model_id: Optional[str] = None) -> float:
    """USD per million input tokens written to a prompt cache (`cost.per_model.<model>.cache_write_input`); the plain input rate when the model has none, so an estimate is never too low."""
    p = ((cfg.get("cost") or {}).get("per_model") or {}).get(model_id or cfg.get("model_id")) or {}
    if p.get("cache_write_input") is not None:
        return float(p["cache_write_input"])
    base = prices(cfg, model_id)
    return float(base["input"]) if base and base["input"] is not None else 0.0
