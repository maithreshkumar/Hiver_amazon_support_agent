from pathlib import Path

from hiver_support.intents.taxonomy import load_approved_taxonomy


def test_approved_taxonomy_has_twelve_unique_intents() -> None:
    taxonomy = load_approved_taxonomy("configs/intents.yaml")
    assert len(taxonomy) == 12
    names = {intent.name for intent in taxonomy}
    assert len(names) == 12
    assert "prime_video_issue" in names
    assert "shipping_promise_failure" not in names
