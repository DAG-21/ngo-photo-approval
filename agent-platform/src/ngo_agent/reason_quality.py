"""Is a written reason informative enough to learn from, and removal of personal contact details before any text could leave the platform.

`is_low_information` is a deliberately simple, deterministic check: a reason that, once filler words are removed, has fewer than three different content words
("looks good to me", "ok", "fine photo") teaches the AI nothing. It never blocks a decision (a person must always be able to submit); it only keeps such
decisions out of the pool of examples shown to the model and tells the person that specific reasons teach more. It can be wrong in both directions
(a long but empty sentence passes; a short precise one such as "faces hidden" fails), so it is a filter on learning, not a judgement of the person.
"""

from __future__ import annotations

import re
from typing import Optional

MIN_CONTENT_WORDS = 3
_FILLER = frozenset("""good nice ok okay fine great looks look looking photo picture image pic yes no approved approve rejected reject bad correct right wrong
the is a an it this that very so quite seems seem acceptable not are was be and or to of for too just all i we think agree disagree ai same as alright decent
perfect best worst poor thing things stuff one me my our you your""".split())
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?!\w)")


def content_words(text: Optional[str]) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z']*", (text or "").lower()) if w not in _FILLER and len(w) > 2}


def is_low_information(text: Optional[str]) -> bool:
    """True when the reason has fewer than three different content words."""
    return len(content_words(text)) < MIN_CONTENT_WORDS


def redact(text: Optional[str]) -> str:
    """Replace e-mail addresses, web links and phone-number-like digit runs. Names cannot be found reliably this way: anything sent to a cloud model
    from free text still needs the NGO's authorisation."""
    t = _EMAIL.sub("[email]", text or "")
    t = _URL.sub("[link]", t)
    return _PHONE.sub("[phone]", t)
