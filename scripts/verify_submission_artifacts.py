from __future__ import annotations

import argparse
import json
import sys

from hiver_support.submission import ArtifactVerificationError, verify_submission_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify all submission artifacts and their hashes")
    parser.add_argument("--manifest", default="artifacts/manifest.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    try:
        result = verify_submission_artifacts(args.manifest, args.root)
    except ArtifactVerificationError as exc:
        print(f"ARTIFACT VERIFICATION FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
