from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import sys

from hiver_support.submission import ArtifactVerificationError, verify_submission_artifacts


REQUIRED_IMPORTS = (
    "accelerate", "joblib", "langdetect", "numpy", "pandas", "pyarrow",
    "safetensors", "sklearn", "torch", "transformers", "yaml",
)


def run_preflight() -> dict[str, object]:
    if sys.version_info < (3, 11):
        raise ArtifactVerificationError(
            f"Python 3.11 or newer is required; found {platform.python_version()}"
        )
    missing = []
    versions: dict[str, str] = {}
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "installed"))
        except ImportError:
            missing.append(name)
    if missing:
        raise ArtifactVerificationError(
            "Missing Python packages: " + ", ".join(missing)
            + ". Run python -m pip install -r requirements-lock.txt"
        )
    lock_mismatches = []
    locked_versions: dict[str, str] = {}
    for raw in open("requirements-lock.txt", encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        distribution, expected = line.split("==", 1)
        actual = importlib.metadata.version(distribution)
        locked_versions[distribution] = actual
        if actual.split("+", 1)[0] != expected:
            lock_mismatches.append(f"{distribution}: expected {expected}, found {actual}")
    if lock_mismatches:
        raise ArtifactVerificationError(
            "Installed packages do not match requirements-lock.txt: "
            + "; ".join(lock_mismatches)
        )
    artifact_result = verify_submission_artifacts()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "locked_distributions": locked_versions,
        "artifacts": artifact_result,
    }


def main() -> None:
    try:
        result = run_preflight()
    except ArtifactVerificationError as exc:
        print(f"PREFLIGHT FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2))
    print("READY FOR HEADLINE REPRODUCTION")


if __name__ == "__main__":
    main()
