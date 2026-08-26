#!/usr/bin/env python
"""Load data/chunks.json into Postgres. Embeddings stay NULL at this stage.

    uv run scripts/load_chunks.py

Idempotent: re-running upserts on chunk_uid, so you can iterate on the
chunker and reload without dropping the database.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.json"
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
DSN = os.environ["DATABASE_URL"]

UPSERT = """
INSERT INTO chunks (chunk_uid, corpus, doc_path, breadcrumb, heading_level,
                    content, word_count, has_code, identifiers)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT (chunk_uid) DO UPDATE SET
    corpus        = EXCLUDED.corpus,
    doc_path      = EXCLUDED.doc_path,
    breadcrumb    = EXCLUDED.breadcrumb,
    heading_level = EXCLUDED.heading_level,
    content       = EXCLUDED.content,
    word_count    = EXCLUDED.word_count,
    has_code      = EXCLUDED.has_code,
    identifiers   = EXCLUDED.identifiers,
    embedding     = NULL
"""


async def main() -> int:
    if not CHUNKS.exists():
        print(f"missing {CHUNKS} — run scripts/run_ingest.py first")
        return 1

    records = json.loads(CHUNKS.read_text())
    conn = await asyncpg.connect(DSN)
    try:
        stale = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE chunk_uid <> ALL($1::text[])",
            [r["chunk_uid"] for r in records],
        )
        await conn.executemany(
            UPSERT,
            [
                (
                    r["chunk_uid"],
                    r["corpus"],
                    r["doc_path"],
                    r["breadcrumb"],
                    r["heading_level"],
                    r["content"],
                    r["word_count"],
                    r["has_code"],
                    r["identifiers"],
                )
                for r in records
            ],
        )
        total = await conn.fetchval("SELECT count(*) FROM chunks")
        embedded = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
        )
    finally:
        await conn.close()

    print(f"loaded     {len(records)}")
    print(f"in table   {total}")
    print(f"embedded   {embedded}")
    if stale:
        print(f"\n{stale} rows no longer in chunks.json (chunker changed).")
        print(
            "Clear them with: DELETE FROM chunks WHERE embedding IS NULL "
            "AND chunk_uid NOT IN (...);  or docker compose down -v"
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
