from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 42

_WHITESPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
_PAYMENT = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_ORDER = re.compile(
    r"\b(?:order\s*(?:id|#|number)?\s*[:#-]?\s*[A-Z0-9-]{8,24}|\d{3}-\d{7}-\d{7})\b",
    re.IGNORECASE,
)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF]")


def normalize_text(text: object) -> str:
    """Apply loss-minimizing normalization; never use this as raw evidence."""
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKC", value)
    return _WHITESPACE.sub(" ", value).strip()


def mask_pii(text: object) -> str:
    value = normalize_text(text)
    value = _EMAIL.sub("[EMAIL]", value)
    value = _PHONE.sub("[PHONE]", value)
    value = _PAYMENT.sub("[PAYMENT_NUMBER]", value)
    return _ORDER.sub("[POSSIBLE_ORDER_ID]", value)


@lru_cache(maxsize=100_000)
def detect_language(text: str) -> str:
    """Return an ISO-639-1 guess, or ``und`` when text is too short/uncertain."""
    normalized = normalize_text(text)
    if len(re.findall(r"[^\W\d_]", normalized, flags=re.UNICODE)) < 8:
        return "und"
    try:
        return detect(normalized)
    except LangDetectException:
        return "und"


def text_metadata(text: object) -> dict[str, object]:
    raw = "" if text is None else str(text)
    normalized = normalize_text(raw)
    masked = mask_pii(normalized)
    return {
        "raw_text": raw,
        "normalized_text": normalized,
        "model_text": masked,
        "url_count": len(_URL.findall(normalized)),
        "mention_count": len(_MENTION.findall(normalized)),
        "has_possible_pii": masked != normalized,
        "character_count": len(normalized),
        "token_count": len(normalized.split()),
        "emoji_present": bool(_EMOJI.search(normalized)),
        "question_present": "?" in normalized,
    }
