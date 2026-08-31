#!/usr/bin/env python
"""Ingest TechQA-RAG-Eval as a second corpus.

    uv run --group ingest scripts/ingest_techqa.py [--limit N]

Writes data/techqa_chunks.json (same shape as chunks.json, corpus='techqa')
and evals/answers/techqa_questions.json.

Why pooling matters
-------------------
Each row ships only its own gold document, so retrieval within a row is
trivial. Pooling all 496 unique documents into one index makes it a real task:
find the right document among 496.

Pooling also rescues the 300 impossible questions, which carry no contexts of
their own. Asked against the pooled corpus, retrieval still returns its five
most similar documents -- confident-looking, non-answering results. That is
the HARD refusal case, and it is what calibrates the threshold: the system
must decline despite plausible retrieval.

Known labelling risk, stated plainly because it belongs in the writeup: a
question marked impossible for its own row might be answerable from some
other row's document in the pool. Spot-check a sample before publishing a
refusal number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingest.chunk import MAX_WORDS, hard_split
from src.ingest.identifiers import extract

URL = "hf://datasets/nvidia/TechQA-RAG-Eval/train.json"
CHUNKS_OUT = ROOT / "data" / "techqa_chunks.json"
QUESTIONS_OUT = ROOT / "evals" / "answers" / "techqa_questions.json"
CORPUS = "techqa"

# One document is 27k words -- an outlier that would dominate the corpus and
# blow the embedding budget. Truncate rather than drop: its gold answer is
# almost certainly near the top of the article.
MAX_DOC_WORDS = 4000

TITLE_RE = re.compile(r"^Title:\s*(.+)$", re.MULTILINE)


def uid(filename: str, index: int) -> str:
    return hashlib.sha1(f"{CORPUS}::{filename}::{index}".encode()).hexdigest()[:16]


def split_doc(text: str) -> list[str]:
    """Paragraph-aware split, with a word-count fallback.

    hard_split works on blank-line paragraph boundaries. Support articles
    scraped to plain text often have none -- one 3,000-word block splits into
    exactly one piece and silently blows the ceiling. The fallback catches it.
    """
    pieces = hard_split(text) if len(text.split()) > MAX_WORDS else [text]
    out: list[str] = []
    for piece in pieces:
        words = piece.split()
        if len(words) <= MAX_WORDS:
            out.append(piece)
            continue
        step = MAX_WORDS - 60  # ~60 words of overlap across forced cuts
        for i in range(0, len(words), step):
            out.append(" ".join(words[i : i + MAX_WORDS]))
    return out


def title_of(text: str, filename: str) -> str:
    m = TITLE_RE.search(text[:500])
    return m.group(1).strip()[:120] if m else filename


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap questions (embedding quota is 1000/day)",
    )
    args = ap.parse_args()

    df = pd.read_json(URL)
    if args.limit:
        # Keep the answerable/impossible balance when sampling.
        imp = df[df.is_impossible].head(args.limit // 3)
        ans = df[~df.is_impossible].head(args.limit - len(imp))
        df = pd.concat([ans, imp]).sample(frac=1, random_state=0)
        print(
            f"limited to {len(df)} rows "
            f"({int((~df.is_impossible).sum())} answerable, "
            f"{int(df.is_impossible.sum())} impossible)"
        )

    docs: dict[str, str] = {}
    for contexts in df["contexts"]:
        for c in contexts:
            docs.setdefault(c["filename"], c["text"])

    records: list[dict] = []
    gold: dict[str, list[str]] = {}
    truncated = 0

    for filename, text in sorted(docs.items()):
        title = title_of(text, filename)
        words = text.split()
        if len(words) > MAX_DOC_WORDS:
            text = " ".join(words[:MAX_DOC_WORDS])
            truncated += 1

        pieces = split_doc(text)
        uids = []
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            cid = uid(filename, i)
            uids.append(cid)
            suffix = f" (part {i + 1}/{len(pieces)})" if len(pieces) > 1 else ""
            records.append(
                {
                    "chunk_uid": cid,
                    "corpus": CORPUS,
                    "doc_path": filename,
                    "breadcrumb": f"TechQA > {title}{suffix}",
                    "heading_level": 1,
                    "content": piece,
                    "word_count": len(piece.split()),
                    "has_code": "```" in piece,
                    "identifiers": extract(piece),
                }
            )
        gold[filename] = uids

    questions = []
    for _, row in df.iterrows():
        names = [c["filename"] for c in row["contexts"]]
        questions.append(
            {
                "id": row["id"],
                "query": row["question"],
                "reference_answer": row["answer"],
                "is_impossible": bool(row["is_impossible"]),
                "gold_docs": names,
                # Any chunk of the gold document counts as a hit: the document is
                # the labelled unit, not our chunk boundaries.
                "gold_chunk_uids": [u for n in names for u in gold.get(n, [])],
            }
        )

    CHUNKS_OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    QUESTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    QUESTIONS_OUT.write_text(json.dumps(questions, indent=2, ensure_ascii=False))

    counts = [r["word_count"] for r in records]
    n_imp = sum(q["is_impossible"] for q in questions)
    multi = sum(1 for uids in gold.values() if len(uids) > 1)

    print(f"\ndocuments        {len(docs)}  (truncated {truncated})")
    print(f"chunks           {len(records)}  ({multi} documents split into several)")
    print(
        f"words min/med/max  {min(counts)} / "
        f"{int(statistics.median(counts))} / {max(counts)}"
    )
    print(
        f"questions        {len(questions)}  "
        f"({len(questions) - n_imp} answerable, {n_imp} impossible)"
    )
    print(
        f"\nembedding cost: {len(records)} chunks + {len(questions)} "
        f"queries = {len(records) + len(questions)} calls"
    )
    print(
        "free tier is ~1000/day -- split across days if needed, the cache "
        "makes re-runs free"
    )
    print(f"\n-> {CHUNKS_OUT}\n-> {QUESTIONS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
