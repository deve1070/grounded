#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.ingest.embed import embed_cached, get_embedder
from src.retrieval.search import (
    detect_identifiers,
    load_vocab,
    search,
)

QUESTIONS = ROOT / "evals" / "answers" / "techqa_questions.json"
RESULTS = ROOT / "evals" / "results" / "threshold_calibration.json"
DSN = os.environ["DATABASE_URL"]
K = 5


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 5),
        "p25": round(ordered[len(ordered) // 4], 5),
        "median": round(statistics.median(ordered), 5),
        "p75": round(ordered[3 * len(ordered) // 4], 5),
        "max": round(ordered[-1], 5),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no-vector",
        action="store_true",
        help="skip query embedding entirely (costs no quota)",
    )
    ap.add_argument("--corpus", default="techqa")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS.read_text())
    conn = await asyncpg.connect(DSN)

    try:
        n_chunks = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE corpus = $1", args.corpus
        )
        n_emb = await conn.fetchval(
            "SELECT count(embedding) FROM chunks WHERE corpus = $1", args.corpus
        )
        if not n_chunks:
            print(f"no '{args.corpus}' chunks loaded -- run load_chunks first")
            return 1
        print(
            f"corpus '{args.corpus}': {n_chunks} chunks "
            f"({n_emb} embedded), {len(questions)} questions"
        )

        vocab = await load_vocab(conn, args.corpus)

        vectors: dict[str, str] = {}
        if args.no_vector:
            print("--no-vector: lexical + exact lanes only\n")
        elif n_emb == 0:
            print("no embeddings for this corpus -- vector lane skipped\n")
        else:
            arr = await embed_cached(
                get_embedder(),
                [q["query"] for q in questions],
                task_type="RETRIEVAL_QUERY",
                label="queries",
            )
            vectors = {
                q["query"]: "[" + ",".join(f"{x:.6f}" for x in v) + "]"
                for q, v in zip(questions, arr)
            }
            print()

        rows = []
        for q in questions:
            hits = await search(
                conn,
                q["query"],
                vectors.get(q["query"]),
                vocab,
                k=K,
                corpus=args.corpus,
            )
            gold = set(q["gold_chunk_uids"])
            rows.append(
                {
                    "id": q["id"],
                    "impossible": q["is_impossible"],
                    "top_score": hits[0].score if hits else 0.0,
                    "gold_retrieved": bool(gold & {h.chunk_uid for h in hits}),
                    "n_hits": len(hits),
                    "idents": detect_identifiers(q["query"], vocab),
                }
            )
    finally:
        await conn.close()

    answerable = [r for r in rows if not r["impossible"]]
    impossible = [r for r in rows if r["impossible"]]

    hit = sum(r["gold_retrieved"] for r in answerable)
    recall = hit / len(answerable) if answerable else 0.0
    empty = sum(1 for r in rows if r["n_hits"] == 0)
    with_idents = sum(1 for r in rows if r["idents"])

    print("=" * 62)
    print("RETRIEVAL ON ANSWERABLE QUESTIONS")
    print("=" * 62)
    print(f"recall@{K}: {recall:.3f}  ({hit}/{len(answerable)})")
    print(f"queries returning nothing: {empty}/{len(rows)}")
    print(f"queries with a detected identifier: {with_idents}/{len(rows)}")
    print(
        "Long forum-post queries against a pooled corpus -- harder, and a"
        "\nmore honest test than single-token identifier lookups."
    )

    a_scores = [r["top_score"] for r in answerable]
    i_scores = [r["top_score"] for r in impossible]
    print("\n" + "=" * 62)
    print("TOP-SCORE DISTRIBUTIONS")
    print("=" * 62)
    print(f"{'':12}{'n':>5}{'min':>9}{'p25':>9}{'median':>9}{'p75':>9}{'max':>9}")
    for name, vals in (("answerable", a_scores), ("impossible", i_scores)):
        s = summarize(vals)
        if s:
            print(
                f"{name:12}{s['n']:>5}{s['min']:>9.4f}{s['p25']:>9.4f}"
                f"{s['median']:>9.4f}{s['p75']:>9.4f}{s['max']:>9.4f}"
            )

    sweep = []
    for t in sorted({round(r["top_score"], 4) for r in rows}):
        kept = sum(1 for r in answerable if r["top_score"] >= t)
        refused = sum(1 for r in impossible if r["top_score"] < t)
        sweep.append(
            {
                "threshold": t,
                "answerable_kept": round(kept / len(answerable), 3),
                "impossible_refused": round(refused / len(impossible), 3),
                "balanced": round(
                    (kept / len(answerable) + refused / len(impossible)) / 2, 3
                ),
            }
        )

    best = max(sweep, key=lambda s: s["balanced"])
    print("\n" + "=" * 62)
    print("THRESHOLD SWEEP")
    print("=" * 62)
    print(
        f"{'threshold':>10}{'answerable kept':>18}"
        f"{'impossible refused':>20}{'balanced':>10}"
    )
    step = max(1, len(sweep) // 15)
    for s in sweep[::step]:
        print(
            f"{s['threshold']:>10.4f}{s['answerable_kept']:>18.3f}"
            f"{s['impossible_refused']:>20.3f}{s['balanced']:>10.3f}"
        )

    print(
        f"\nbest balanced: MIN_SCORE = {best['threshold']:.4f}  "
        f"(keeps {best['answerable_kept']:.1%} answerable, "
        f"refuses {best['impossible_refused']:.1%} impossible)"
    )

    if best["balanced"] < 0.62:
        print(
            "\nThe distributions overlap: no retrieval-score cutoff cleanly\n"
            "separates answerable from unanswerable here. Keep Gate 1 loose\n"
            "and let the citation gate carry refusal. Publish this -- it is\n"
            "the measured argument for a second gate."
        )

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "corpus": args.corpus,
                "k": K,
                "lanes": "lexical+exact" if not vectors else "lexical+vector+exact",
                "n_chunks": n_chunks,
                "recall_at_k_answerable": round(recall, 3),
                "empty_result_queries": empty,
                "score_distributions": {
                    "answerable": summarize(a_scores),
                    "impossible": summarize(i_scores),
                },
                "best": best,
                "sweep": sweep,
                "note": (
                    "RRF scores scale with lane count; a threshold from one lane "
                    "configuration does not transfer to another."
                ),
            },
            indent=2,
        )
    )
    print(f"\n-> {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
