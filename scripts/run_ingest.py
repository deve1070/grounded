#!/usr/bin/env python
"""Offline ingestion. No database, no embeddings, no API calls.

    uv run --group ingest scripts/run_ingest.py

Outputs:
    data/chunks.json          machine-readable, feeds embedding + load
    data/chunks_preview.md
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest import chunk as chunker
from src.ingest import identifiers as ident
from src.ingest import mdx

CORPUS = ROOT / "corpus" / "chatwoot"
OUT_DIR = ROOT / "data"
CORPUS_NAME = "chatwoot"


def uid(doc_path: str, crumb: str, index: int) -> str:
    raw = f"{doc_path}::{crumb}::{index}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def main() -> int:
    files = sorted(CORPUS.rglob("*.mdx"))
    if not files:
        print(f"No .mdx under {CORPUS} — run the vendoring step first.")
        return 1

    records: list[dict] = []
    skipped: list[str] = []

    for path in files:
        try:
            doc = mdx.parse_file(path, CORPUS)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{path.relative_to(CORPUS)}: {exc}")
            continue

        for i, ch in enumerate(chunker.chunk_doc(doc)):
            content = mdx.restore_code(ch.content, doc.code_blocks).strip()
            if not content:
                continue
            records.append(
                {
                    "chunk_uid": uid(ch.doc_path, ch.crumb_text, i),
                    "corpus": CORPUS_NAME,
                    "doc_path": ch.doc_path,
                    "breadcrumb": ch.crumb_text,
                    "heading_level": ch.heading_level,
                    "content": content,
                    "word_count": len(content.split()),
                    "has_code": "```" in content,
                    "identifiers": ident.extract(content),
                }
            )

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "chunks.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False)
    )

    counts = [r["word_count"] for r in records]
    all_ids = Counter(i for r in records for i in r["identifiers"])

    lines = ["# Chunk preview", ""]
    for r in records:
        body = r["content"]
        preview = body if len(body) < 700 else body[:700] + "\n…[truncated]"
        lines += [
            f"## {r['breadcrumb']}",
            f"`{r['doc_path']}` · {r['word_count']} words · "
            f"code={r['has_code']} · ids={', '.join(r['identifiers']) or '—'}",
            "",
            "```text",
            preview,
            "```",
            "",
        ]
    (OUT_DIR / "chunks_preview.md").write_text("\n".join(lines))

    print(f"files parsed     {len(files) - len(skipped)}/{len(files)}")
    print(f"chunks           {len(records)}")
    print(
        f"words  min/med/max  {min(counts)} / "
        f"{int(statistics.median(counts))} / {max(counts)}"
    )
    print(
        f"under {chunker.MIN_WORDS}w      {sum(c < chunker.MIN_WORDS for c in counts)}"
    )
    print(
        f"over  {chunker.MAX_WORDS}w      "
        f"{sum(c > chunker.MAX_WORDS for c in counts)}  (code-heavy is fine)"
    )
    print(f"with code        {sum(r['has_code'] for r in records)}")
    print(f"unique ids       {len(all_ids)}")
    print(f"ids once only    {sum(1 for _, n in all_ids.items() if n == 1)}")
    if skipped:
        print("\nskipped:")
        for s in skipped:
            print("  ", s)
    print(f"\n→ {OUT_DIR / 'chunks.json'}\n→ {OUT_DIR / 'chunks_preview.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
