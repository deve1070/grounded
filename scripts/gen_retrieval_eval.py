#!/usr/bin/env python
"""Build a retrieval eval set from identifiers that appear in exactly one chunk.

    uv run --group ingest scripts/gen_retrieval_eval.py

No LLM, no labeling: if `SIDEKIQ_CONCURRENCY` occurs in one chunk only, that
chunk is the gold answer for any question about it, by construction.

Two query styles per identifier, because they stress retrieval differently:

  bare      "SIDEKIQ_CONCURRENCY"
            What users actually type. Trivial for exact match, near-useless
            for embeddings — this is where the hybrid gap shows.

  natural   "What does SIDEKIQ_CONCURRENCY configure?"
            Surrounding prose gives the vector lane something to work with,
            so the gap should narrow. If it doesn't, that is a finding.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "chunks.json"
OUT = ROOT / "evals" / "retrieval" / "identifier_questions.json"

TEMPLATES = [
    "What does {} configure?",
    "How do I set {}?",
    "What is {} used for?",
]


def main() -> int:
    records = json.loads(CHUNKS.read_text())

    where: dict[str, set[str]] = defaultdict(set)
    for r in records:
        for ident in r["identifiers"]:
            where[ident].add(r["chunk_uid"])

    unique = {i: next(iter(u)) for i, u in where.items() if len(u) == 1}

    questions = []
    for n, (ident, uid) in enumerate(sorted(unique.items())):
        questions.append(
            {
                "id": f"bare-{ident}",
                "style": "bare",
                "query": ident,
                "identifier": ident,
                "gold_chunk_uid": uid,
            }
        )
        questions.append(
            {
                "id": f"nat-{ident}",
                "style": "natural",
                "query": TEMPLATES[n % len(TEMPLATES)].format(ident),
                "identifier": ident,
                "gold_chunk_uid": uid,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(questions, indent=2))

    print(f"identifiers total    {len(where)}")
    print(f"single-chunk only    {len(unique)}")
    print(f"questions written    {len(questions)}")
    print(f"\n-> {OUT}")
    print("\nSpot-check a few. If an identifier appears once but the chunk")
    print("only mentions it in passing, the 'gold' answer is wrong and the")
    print("question should be dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
