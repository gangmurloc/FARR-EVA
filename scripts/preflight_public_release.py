"""Fail on common accidental disclosures in the compact public repository."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}
FORBIDDEN_DIRECTORIES = {
    "analysis_logs",
    "data",
    "drafts",
    "outputs",
    "submission_eswa",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PATTERNS = {
    # Split the literal so this scanner does not flag its own source.
    "machine_home": re.compile("/" + r"home/[^/\s]+/"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "secret_assignment": re.compile(
        r"(?i)(api[_-]?key|access[_-]?token|hf_token|openai_api_key)\s*[:=]\s*['\"][^'\"]+"
    ),
}


def files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not (set(path.parts) & IGNORED_PARTS)
    ]


def main() -> None:
    findings: list[str] = []
    for name in FORBIDDEN_DIRECTORIES:
        if (ROOT / name).exists():
            findings.append(f"forbidden directory present: {name}")
    for path in files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size > 10 * 1024 * 1024:
            findings.append(f"file exceeds 10 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"non-UTF-8 text-like file: {relative}")
            continue
        for kind, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{kind}: {relative}")
    status = "PASS" if not findings else "FAIL"
    print(f"public-release preflight: {status}")
    print(f"files scanned: {len(files())}")
    for finding in findings:
        print(f"- {finding}")
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
