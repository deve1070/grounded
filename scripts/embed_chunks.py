#!/usr/bin/env python
"""Embed the corpus once, offline, and commit the result.

    uv run --group ingest scripts/embed_chunks.py

Writes data/embeddings.npy (~600 KB at 187x768 float32) and pushes the
vectors into Postgres.

The .npy is committed deliberately. It means the deployed demo boots with a
populated index and zero API calls, so neither a wiped database nor an
exhausted quota can take the flagship demo offline.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.ingest.embed import embed_cached, get_embedder

CHUNKS = ROOT / "data" / "chunks.json"
# VECTORS = ROOT / "data" / "embeddings.npy"
# ORDER = ROOT / "data" / "embeddings_uids.json"
DSN = os.environ["DATABASE_URL"]
# VECTORS = CHUNKS.with_name(CHUNKS.stem.replace("chunks", "embeddings") + ".npy")
# ORDER = CHUNKS.with_name(CHUNKS.stem + "_uids.json")

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", default="data/chunks.json")
args = ap.parse_args()
CHUNKS = ROOT / args.chunks
   
VECTORS = CHUNKS.with_name(CHUNKS.stem.replace("chunks", "embeddings") + ".npy")
ORDER   = CHUNKS.with_name(CHUNKS.stem + "_uids.json")


def embed_text(record: dict) -> str:
    """Breadcrumb is prepended, not just stored.

    Headings in this corpus are things like "Sent" and "Screencast
    Walkthrough" -- meaningless alone. The breadcrumb is what makes a chunk
    locatable in vector space as well as readable in a citation.
    """
    return f"{record['breadcrumb']}\n\n{record['content']}"


async def main() -> int:
    records = json.loads(CHUNKS.read_text())
    embedder = get_embedder()
    print(f"model: {embedder.model}  chunks: {len(records)}")

    vectors = await embed_cached(
        embedder,
        [embed_text(r) for r in records],
        task_type="RETRIEVAL_DOCUMENT",
        label="corpus",
    )

    uids = [r["chunk_uid"] for r in records]
    np.save(VECTORS, vectors)
    ORDER.write_text(json.dumps(uids, indent=2))
    print(
        f"saved {vectors.shape} -> {VECTORS} ({VECTORS.stat().st_size / 1024:.0f} KB)"
    )

    conn = await asyncpg.connect(DSN)
    try:
        await conn.executemany(
            "UPDATE chunks SET embedding = $2::vector WHERE chunk_uid = $1",
            [
                (uid, "[" + ",".join(f"{x:.6f}" for x in vec) + "]")
                for uid, vec in zip(uids, vectors)
            ],
        )
        done = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
        )
        total = await conn.fetchval("SELECT count(*) FROM chunks")
    finally:
        await conn.close()

    print(f"embedded in db: {done}/{total}")
    if done < total:
        print(
            "some rows unembedded -- chunk_uids in the db do not match "
            "chunks.json; re-run scripts/load_chunks.py"
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
