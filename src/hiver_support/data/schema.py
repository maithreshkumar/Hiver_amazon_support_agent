from __future__ import annotations

REQUIRED_COLUMNS = (
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
)


class SchemaError(ValueError):
    """Raised when the source dataset cannot be interpreted safely."""


def validate_columns(columns: list[str] | tuple[str, ...]) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise SchemaError(
            "twcs.csv is missing required columns: " + ", ".join(missing)
        )


def parse_bool(value: object) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def parse_id_list(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]

