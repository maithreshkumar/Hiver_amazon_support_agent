from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hiver_support.submission import (
    ArtifactVerificationError,
    load_artifact_manifest,
    verify_submission_artifacts,
)


def _is_missing_or_pointer(path: Path) -> bool:
    if not path.is_file():
        return True
    return path.stat().st_size <= 1024 and path.read_bytes().startswith(
        b"version https://git-lfs.github.com/spec/v1"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch missing submission artifacts without downloading the raw Twitter dataset"
    )
    parser.add_argument("--manifest", default="artifacts/manifest.json")
    args = parser.parse_args()
    manifest = load_artifact_manifest(args.manifest)
    root = Path(".").resolve()
    lfs_paths: list[str] = []
    unavailable: list[str] = []
    for artifact in manifest["artifacts"]:
        if not artifact.get("required"):
            continue
        relative = str(artifact["path"])
        if not _is_missing_or_pointer(root / relative):
            continue
        source = artifact.get("source", {})
        if source.get("type") == "git_lfs":
            lfs_paths.append(relative.replace("\\", "/"))
        else:
            unavailable.append(relative)
    if lfs_paths:
        print("Fetching Git LFS artifacts: " + ", ".join(lfs_paths))
        try:
            subprocess.run(
                ["git", "lfs", "pull", f"--include={','.join(lfs_paths)}"],
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise SystemExit(
                "Git LFS fetch failed. Install Git LFS, run 'git lfs install', and retry."
            ) from exc
    if unavailable:
        raise SystemExit(
            "Required repository artifacts are missing and have no external fetch source: "
            + ", ".join(unavailable)
            + ". Restore them from the submitted repository revision."
        )
    try:
        result = verify_submission_artifacts(args.manifest)
    except ArtifactVerificationError as exc:
        print(f"FETCHED ARTIFACTS FAILED VERIFICATION\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2))
    print("All required submission artifacts are present and verified.")


if __name__ == "__main__":
    main()
