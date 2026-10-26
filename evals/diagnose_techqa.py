#!/usr/bin/env python
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

QUESTIONS = ROOT / "evals" / "answers" / "techqa_questions.json"
DSN = os.environ["DATABASE_URL"]
CORPUS = "techqa"


async def main() -> int:
    qs = json.loads(QUESTIONS.read_text())
    conn = await asyncpg.connect(DSN)

    # 1. Do the gold uids exist in the table at all?
    gold = {u for q in qs for u in q["gold_chunk_uids"]}
    present = await conn.fetchval(
        "SELECT count(*) FROM chunks WHERE chunk_uid = ANY($1::text[])", list(gold)
    )
    print(f"1. gold uids referenced   {len(gold)}")
    print(f"   found in chunks table  {present}")
    if present == 0:
        print(
            "   -> BROKEN: questions and chunks were generated from "
            "different runs. Re-run ingest_techqa.py, then load_chunks.py."
        )

    # 2. Corpus tag and embedding state.
    row = await conn.fetchrow(
        "SELECT count(*) AS n, count(embedding) AS emb FROM chunks WHERE corpus = $1",
        CORPUS,
    )
    print(f"\n2. chunks with corpus='{CORPUS}'  {row['n']}")
    print(f"   of those, embedded            {row['emb']}")
    if row["emb"] == 0:
        print("   -> vector lane returns nothing; lexical is running alone.")

    # 3. Does plainto_tsquery survive a 44-word forum post?
    #    It ANDs every lexeme together, so long queries match nothing.
    q = qs[0]["query"]
    for fn in ("plainto_tsquery", "websearch_to_tsquery"):
        n = await conn.fetchval(
            f"SELECT count(*) FROM chunks WHERE corpus = $1 "
            f"AND tsv @@ {fn}('english', $2)",
            CORPUS,
            q,
        )
        print(f"\n3. {fn:22} matches {n} chunks for a real question")

    # 4. What does lexical actually return for a question, vs its gold?
    hits = await conn.fetch(
        "SELECT chunk_uid, breadcrumb FROM chunks WHERE corpus = $1 "
        "AND tsv @@ plainto_tsquery('english', $2) "
        "ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $2)) DESC "
        "LIMIT 5",
        CORPUS,
        q,
    )
    print(f"\n4. question: {q[:90]}...")
    print(f"   gold uids: {qs[0]['gold_chunk_uids']}")
    print(f"   returned : {[h['chunk_uid'] for h in hits] or 'NOTHING'}")

    await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
