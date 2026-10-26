from __future__ import annotations

import re

BACKTICKED_RE = re.compile(r"`([^`\n]{2,80})`")
SCREAMING_RE = re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b")
KEBAB_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$")
DOTTED_RE = re.compile(r"^[a-z_][a-z0-9_]*(?:\.[a-z0-9_]+)+$")


BLOCKLIST = {
    "PASSWORD",
    "EMAIL",
    "USERNAME",
    "TRUE",
    "FALSE",
    "NULL",
    "NONE",
    "NOTE",
    "WARNING",
    "INFO",
    "TIP",
    "ERROR",
    "TODO",
    "JSON",
    "YAML",
    "HTTP",
    "HTTPS",
    "HTML",
    "CSS",
    "SQL",
    "URL",
    "URI",
    "GET",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
    "API",
    "UUID",
    "ID",
    "OK",
    "README",
    "LICENSE",
    "MIT",
    "AND",
    "OR",
    "NOT",
    "ALL",
    "ANY",
}

MIN_BARE_LENGTH = 9


def _acceptable(token: str) -> bool:
    if token.upper() in BLOCKLIST:
        return False
    if "_" in token:
        return True
    return len(token) >= MIN_BARE_LENGTH


def extract(text: str) -> list[str]:
    found: set[str] = set()

    # SCREAMING_SNAKE anywhere — including inside code blocks.
    for token in SCREAMING_RE.findall(text):
        if _acceptable(token):
            found.add(token)

    # kebab-case and dotted names only when backticked. Bare kebab-case would
    # match ordinary prose like "self-hosted" on every other line.
    for raw in BACKTICKED_RE.findall(text):
        token = raw.strip()
        if " " in token:
            continue
        if KEBAB_RE.match(token) or DOTTED_RE.match(token):
            if _acceptable(token):
                found.add(token)

    return sorted(found)
