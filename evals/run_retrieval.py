#!/usr/bin/env python
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import asyncpg
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.ingest.embed import embed_cached, get_embedder  # noqa: E402

QUESTIONS = ROOT / "evals" / "retrieval" / "identifier_questions.json"
RESULTS = ROOT / "evals" / "results" / "retrieval_baseline.json"

# Fail loudly rather than silently connecting to whatever is on a default port.
DSN = os.environ["DATABASE_URL"]

K = 5
RRF_K = 60
IDENT_BOOST = 2.0

VOCAB_SQL = "SELECT DISTINCT unnest(identifiers) AS ident FROM chunks"

LEXICAL_SQL = """
SELECT chunk_uid
FROM chunks
WHERE tsv @@ plainto_tsquery('english', $1)
ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $1)) DESC
LIMIT $2
"""

EXACT_SQL = """
SELECT chunk_uid
FROM chunks
WHERE identifiers && $1::text[]
LIMIT $2
"""

VECTOR_SQL = """
SELECT chunk_uid
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT $2
"""

TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-]+")


def detect_identifiers(query: str, vocab: dict[str, str]) -> list[str]:
    """Match query tokens against known corpus identifiers.

    Deterministic and free -- no LLM call, which matters under free-tier
    quota. Zero false positives by construction: a token that was never
    indexed cannot match.
    """
    hits: set[str] = set()
    for raw in TOKEN_RE.findall(query):
        token = raw.strip(".-_")
        if token and token.lower() in vocab:
            hits.add(vocab[token.lower()])
    return sorted(hits)


def rrf(lists: list[tuple[list[str], float]]) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked, weight in lists:
        for rank, uid in enumerate(ranked, start=1):
            scores[uid] += weight / (RRF_K + rank)
    return [
        uid for uid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]


async def main() -> int:
    if not QUESTIONS.exists():
        print("run scripts/gen_retrieval_eval.py first")
        return 1

    questions = json.loads(QUESTIONS.read_text())
    conn = await asyncpg.connect(DSN)

    hits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen: dict[str, int] = defaultdict(int)
    detected: dict[str, int] = defaultdict(int)
    misses: list[dict] = []

    try:
        vocab = {r["ident"].lower(): r["ident"] for r in await conn.fetch(VOCAB_SQL)}
        if not vocab:
            print("chunks table is empty -- run scripts/load_chunks.py")
            return 1

        has_vectors = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL"
        )
        qvecs: dict[str, str] = {}
        if has_vectors:
            texts = [q["query"] for q in questions]
            arr = await embed_cached(
                get_embedder(), texts, task_type="RETRIEVAL_QUERY", label="queries"
            )
            qvecs = {
                t: "[" + ",".join(f"{x:.6f}" for x in v) + "]"
                for t, v in zip(texts, arr)
            }
        else:
            print(
                "no embeddings in db -- vector lane skipped "
                "(run scripts/embed_chunks.py)\n"
            )

        for q in questions:
            gold, style = q["gold_chunk_uid"], q["style"]
            seen[style] += 1
            seen["all"] += 1

            lex = [r["chunk_uid"] for r in await conn.fetch(LEXICAL_SQL, q["query"], K)]

            idents = detect_identifiers(q["query"], vocab)
            if idents:
                detected[style] += 1
                detected["all"] += 1
            exact = (
                [r["chunk_uid"] for r in await conn.fetch(EXACT_SQL, idents, K)]
                if idents
                else []
            )

            vec = (
                [
                    r["chunk_uid"]
                    for r in await conn.fetch(VECTOR_SQL, qvecs[q["query"]], K)
                ]
                if qvecs
                else []
            )

            hybrid = rrf([(lex, 1.0), (vec, 1.0)])[:K]
            fused = rrf([(lex, 1.0), (vec, 1.0), (exact, IDENT_BOOST)])[:K]

            for name, ranked in (
                ("lexical", lex[:K]),
                ("vector", vec[:K]),
                ("hybrid", hybrid),
                ("exact", exact[:K]),
                ("fused", fused),
            ):
                if gold in ranked:
                    hits[name][style] += 1
                    hits[name]["all"] += 1
                elif name == "fused":
                    misses.append(
                        {
                            "query": q["query"],
                            "style": style,
                            "identifier": q["identifier"],
                            "detected": idents,
                            "gold": gold,
                        }
                    )
    finally:
        await conn.close()

    styles = [s for s in ("all", "bare", "natural") if s in seen]
    lanes = ("lexical", "vector", "hybrid", "exact", "fused")
    table = {
        name: {s: round(hits[name][s] / seen[s], 3) for s in styles} for name in lanes
    }
    detect_rate = {s: round(detected[s] / seen[s], 3) for s in styles}

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "k": K,
                "n": seen["all"],
                "rrf_k": RRF_K,
                "identifier_boost": IDENT_BOOST,
                "identifier_detection_rate": detect_rate,
                "recall": table,
                "note": (
                    "exact and fused are 1.0 BY CONSTRUCTION: gold is defined as the "
                    "chunk containing identifier X, and the exact lane matches on that "
                    "same array. Only lexical and vector are independent measurements."
                ),
                "misses": misses,
            },
            indent=2,
        )
    )

    header = f"{'config':10}" + "".join(f"{s:>9}" for s in styles)
    print(f"n = {seen['all']}   k = {K}\n")
    print(header)
    for name, row in table.items():
        print(f"{name:10}" + "".join(f"{row[s]:>9.3f}" for s in styles))
    print(f"\n{'detected':10}" + "".join(f"{detect_rate[s]:>9.3f}" for s in styles))
    print(f"\nfused misses: {len(misses)}  ->  {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
