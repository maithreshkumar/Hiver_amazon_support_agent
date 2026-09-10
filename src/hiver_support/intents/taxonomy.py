from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from hiver_support.errors import HumanApprovalRequired


@dataclass(frozen=True, slots=True)
class IntentDefinition:
    name: str
    description: str
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    common_confusions: list[str]


def load_approved_taxonomy(path: str | Path = "configs/intents.yaml") -> list[IntentDefinition]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if payload.get("status") != "approved_by_human":
        raise HumanApprovalRequired(
            f"{source} is {payload.get('status', 'missing status')!r}; record explicit human approval "
            "and 8-15 complete intent definitions before weak labeling or classifier training."
        )
    raw_intents = payload.get("intents") or []
    if not 8 <= len(raw_intents) <= 15:
        raise HumanApprovalRequired("Approved taxonomy must contain between 8 and 15 intents.")
    required = {"name", "description", "inclusion_criteria", "exclusion_criteria", "common_confusions"}
    definitions: list[IntentDefinition] = []
    for raw in raw_intents:
        missing = required - set(raw)
        if missing:
            raise HumanApprovalRequired(
                f"Intent {raw.get('name', '<unnamed>')} lacks: {', '.join(sorted(missing))}"
            )
        definitions.append(IntentDefinition(**{key: raw[key] for key in required}))
    names = [item.name for item in definitions]
    if len(set(names)) != len(names):
        raise HumanApprovalRequired("Approved intent names must be unique.")
    return definitions

