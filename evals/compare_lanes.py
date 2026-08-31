#!/usr/bin/env python
"""Lane-by-lane recall on TechQA. The table that goes in the README.

    uv run evals/compare_lanes.py [--corpus techqa]

calibrate_threshold.py measures the fused system. This measures each lane
alone and then the fusion, so the contribution of each is visible instead of
assumed -- the same comparison that showed vector losing everywhere on
Chatwoot yet still lifting the fused score.

Impossible questions are reported separately as "false-hit rate": the share
where retrieval returned something at all. There is no gold chunk for them,
so any confident result is a chance for the generator to hallucinate, and
this is the number Gate 2 has to survive.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.ingest.embed import embed_cached, get_embedder
from src.retrieval.search import (
    CANDIDATES,
    EXACT_SQL,
    IDENT_BOOST,
    LEXICAL_SQL,
    VECTOR_SQL,
    detect_identifiers,
    load_vocab,
    rrf,
)

QUESTIONS = ROOT / "evals" / "answers" / "techqa_questions.json"
RESULTS = ROOT / "evals" / "results" / "techqa_lanes.json"
DSN = os.environ["DATABASE_URL"]
K = 5


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="techqa")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS.read_text())
    conn = await asyncpg.connect(DSN)

    try:
        n_emb = await conn.fetchval(
            "SELECT count(embedding) FROM chunks WHERE corpus = $1", args.corpus
        )
        if not n_emb:
            print("no embeddings for this corpus -- run embed_chunks first")
            return 1

        embedder = get_embedder()
        print(
            f"corpus '{args.corpus}': {n_emb} embedded chunks, model {embedder.model}"
        )

        vocab = await load_vocab(conn, args.corpus)
        arr = await embed_cached(
            embedder,
            [q["query"] for q in questions],
            task_type="RETRIEVAL_QUERY",
            label="queries",
        )
        qvec = {
            q["query"]: "[" + ",".join(f"{x:.6f}" for x in v) + "]"
            for q, v in zip(questions, arr)
        }

        hits: dict[str, int] = defaultdict(int)
        returned: dict[str, int] = defaultdict(int)
        n_ans = n_imp = 0

        for q in questions:
            lex = [
                r["chunk_uid"]
                for r in await conn.fetch(
                    LEXICAL_SQL, q["query"], CANDIDATES, args.corpus
                )
            ]
            vec = [
                r["chunk_uid"]
                for r in await conn.fetch(
                    VECTOR_SQL, qvec[q["query"]], CANDIDATES, args.corpus
                )
            ]
            idents = detect_identifiers(q["query"], vocab)
            exact = (
                [
                    r["chunk_uid"]
                    for r in await conn.fetch(
                        EXACT_SQL, idents, CANDIDATES, args.corpus
                    )
                ]
                if idents
                else []
            )

            lanes = {"lexical": (lex, 1.0), "vector": (vec, 1.0)}
            hybrid = [u for u, _, _ in rrf(lanes)][:K]
            if exact:
                lanes["exact"] = (exact, IDENT_BOOST)
            fused = [u for u, _, _ in rrf(lanes)][:K]

            sets = {
                "lexical": lex[:K],
                "vector": vec[:K],
                "exact": exact[:K],
                "hybrid": hybrid,
                "fused": fused,
            }

            if q["is_impossible"]:
                n_imp += 1
                for name, ranked in sets.items():
                    if ranked:
                        returned[name] += 1
            else:
                n_ans += 1
                gold = set(q["gold_chunk_uids"])
                for name, ranked in sets.items():
                    if gold & set(ranked):
                        hits[name] += 1
    finally:
        await conn.close()

    order = ["lexical", "vector", "exact", "hybrid", "fused"]
    print("\n" + "=" * 58)
    print(f"RECALL@{K} ON {n_ans} ANSWERABLE QUESTIONS")
    print("=" * 58)
    for name in order:
        print(f"{name:10}{hits[name] / n_ans:>8.3f}   ({hits[name]}/{n_ans})")

    print("\n" + "=" * 58)
    print(f"FALSE-HIT RATE ON {n_imp} IMPOSSIBLE QUESTIONS")
    print("=" * 58)
    print(
        "share where retrieval returned anything -- no gold exists, so each\n"
        "is an opportunity to hallucinate that Gate 2 must catch\n"
    )
    for name in order:
        print(f"{name:10}{returned[name] / n_imp:>8.3f}   ({returned[name]}/{n_imp})")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "corpus": args.corpus,
                "k": K,
                "embedding_model": get_embedder().model,
                "n_answerable": n_ans,
                "n_impossible": n_imp,
                "recall": {n: round(hits[n] / n_ans, 3) for n in order},
                "false_hit_rate": {n: round(returned[n] / n_imp, 3) for n in order},
            },
            indent=2,
        )
    )
    print(f"\n-> {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
