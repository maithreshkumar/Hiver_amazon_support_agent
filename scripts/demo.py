from __future__ import annotations

import json

from hiver_support.agent import AmazonSupportAgent
from hiver_support.errors import ConfigurationError


def main() -> None:
    print("Amazon Support Agent\n--------------------")
    try:
        agent = AmazonSupportAgent()
    except ConfigurationError as exc:
        raise SystemExit(f"Cannot start: {exc}") from exc
    while True:
        message = input("\nCustomer > ").strip()
        if message.casefold() in {"quit", "exit"}:
            break
        if message:
            print(json.dumps(agent.handle(message), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

