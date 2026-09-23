"""Verify the published file set and SHA-256 checksums."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED = {"SHA256SUMS"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    recorded = {}
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        assert len(digest) == 64 and relative not in recorded
        recorded[relative] = digest
    actual = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(ROOT).parts
        and "__pycache__" not in p.relative_to(ROOT).parts
        and p.suffix != ".pyc" and p.name not in EXCLUDED
    }
    assert actual == set(recorded), {
        "missing": sorted(set(recorded) - actual),
        "extra": sorted(actual - set(recorded)),
    }
    failures = [relative for relative, digest in recorded.items()
                if sha256(ROOT / relative) != digest]
    assert not failures, failures
    print(f"PASS: {len(recorded)} published files match SHA256SUMS.")


if __name__ == "__main__":
    main()
