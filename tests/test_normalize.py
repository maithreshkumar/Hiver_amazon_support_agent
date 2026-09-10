from hiver_support.data.normalize import detect_language, normalize_text, text_metadata


def test_normalization_preserves_signal() -> None:
    raw = "Great.  Another package lost. Fantastic service 👍\n"
    normalized = normalize_text(raw)
    assert normalized == "Great. Another package lost. Fantastic service 👍"
    assert text_metadata(raw)["emoji_present"] is True


def test_raw_is_preserved_and_pii_is_model_only() -> None:
    raw = "Email me at human@example.com"
    metadata = text_metadata(raw)
    assert metadata["raw_text"] == raw
    assert metadata["normalized_text"] == raw
    assert "[EMAIL]" in metadata["model_text"]


def test_normal_words_are_not_mistaken_for_order_ids() -> None:
    metadata = text_metadata("This experience was disappointing")
    assert metadata["model_text"] == "This experience was disappointing"


def test_language_detection_is_deterministic() -> None:
    assert detect_language("My package was marked delivered but it is not here") == "en"
    assert detect_language("help") == "und"
