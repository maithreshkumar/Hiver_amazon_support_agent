from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from hiver_support.errors import ProviderUnavailable


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderUnavailable(f"Provider request failed for {url}: {exc}") from exc


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ProviderUnavailable("Provider did not return a JSON object") from exc
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as fallback_exc:
            raise ProviderUnavailable("Provider returned malformed or truncated JSON") from fallback_exc
    if not isinstance(value, dict):
        raise ProviderUnavailable("Provider structured output must be a JSON object")
    return value
