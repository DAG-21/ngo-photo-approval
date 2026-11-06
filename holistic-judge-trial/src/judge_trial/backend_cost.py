"""Offline token and cost estimates. Nothing here touches the network.

All figures are ESTIMATES built from the config (tokens per image is an assumption). After the first real call the
measured ``prompt_tokens`` / ``cached_tokens`` in results.jsonl replace them.
"""

from __future__ import annotations

from .config import TrialConfig
from .prompt import BuiltPrompt


def estimate_prefix_tokens(prompt: BuiltPrompt, tokens_per_image: int) -> int:
    """Rough size of the fixed part of the prompt: about 4 characters per token plus a flat cost per example photo."""
    text = len(prompt.system_text) + (len(prompt.examples_intro) if prompt.examples else 0)
    text += sum(len(e.decision_text) + 20 for e in prompt.examples)
    return text // 4 + tokens_per_image * len(prompt.examples)


def caching_expected(prompt: BuiltPrompt, cfg: TrialConfig) -> bool:
    mode = cfg.cache.mode
    return mode == "on" or (mode == "auto" and estimate_prefix_tokens(prompt, cfg.cost.tokens_per_image) >= cfg.cache.min_prefix_tokens)


def estimate_cost(prompt: BuiltPrompt, cfg: TrialConfig, n_photos: int) -> dict:
    c = cfg.cost
    prefix = estimate_prefix_tokens(prompt, c.tokens_per_image)
    per_photo_new = c.tokens_per_image + len(prompt.user_text) // 4
    caching = caching_expected(prompt, cfg)
    out = {
        "prefix_tokens": prefix,
        "per_photo_tokens": per_photo_new,
        "per_photo_output_tokens": c.expected_output_tokens,
        "caching_expected": caching,
        "input_tokens_no_cache": n_photos * (prefix + per_photo_new),
        "input_tokens_with_cache_fresh": prefix + n_photos * per_photo_new if caching else None,
        "input_tokens_with_cache_cached_reads": n_photos * prefix if caching else None,
        "output_tokens": n_photos * c.expected_output_tokens,
        "usd": None,
    }
    if c.input_per_million_usd is None or c.output_per_million_usd is None:
        return out
    m = 1_000_000
    no_cache = out["input_tokens_no_cache"] / m * c.input_per_million_usd + out["output_tokens"] / m * c.output_per_million_usd
    usd = {"without_cache": round(no_cache, 4)}
    if caching and c.cached_input_per_million_usd is not None:
        hours = cfg.cache.ttl_seconds / 3600
        storage = (prefix / m * c.cache_storage_per_million_tokens_per_hour_usd * hours
                   if c.cache_storage_per_million_tokens_per_hour_usd is not None else 0.0)
        usd["with_cache"] = round(
            (prefix + n_photos * per_photo_new) / m * c.input_per_million_usd
            + n_photos * prefix / m * c.cached_input_per_million_usd
            + out["output_tokens"] / m * c.output_per_million_usd + storage, 4)
    out["usd"] = usd
    return out
