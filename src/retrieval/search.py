from __future__ import annotations

import re
from dataclasses import dataclass

import asyncpg

DEFAULT_CORPUS = "chatwoot"
K_DEFAULT = 5
CANDIDATES = 20
RRF_K = 60
IDENT_BOOST = 2.0

TOKEN_RE = re.compile(r"[A-Za-z0-9_.\-]+")

LEXICAL_SQL = """
WITH q AS (
  SELECT array_to_string(
           ARRAY(SELECT quote_literal(l)
                 FROM unnest(tsvector_to_array(to_tsvector('english', $1))) l),
           ' | '
         )::tsquery AS query
)
SELECT chunk_uid, ts_rank_cd(tsv, q.query) AS score
FROM chunks, q
WHERE corpus = $3 AND tsv @@ q.query
ORDER BY score DESC
LIMIT $2
"""

VECTOR_SQL = """
SELECT chunk_uid, 1 - (embedding <=> $1::vector) AS score
FROM chunks
WHERE corpus = $3 AND embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT $2
"""

EXACT_SQL = """
SELECT chunk_uid, cardinality(identifiers) AS breadth
FROM chunks
WHERE corpus = $3 AND identifiers && $1::text[]
ORDER BY breadth ASC
LIMIT $2
"""

HYDRATE_SQL = """
SELECT chunk_uid, doc_path, breadcrumb, content, has_code
FROM chunks
WHERE chunk_uid = ANY($1::text[])
"""

VOCAB_SQL = "SELECT DISTINCT unnest(identifiers) AS ident FROM chunks WHERE corpus = $1"


@dataclass
class Hit:
    chunk_uid: str
    score: float
    breadcrumb: str = ""
    doc_path: str = ""
    content: str = ""
    has_code: bool = False
    lanes: tuple[str, ...] = ()


def detect_identifiers(query: str, vocab: dict[str, str]) -> list[str]:
    """Match query tokens against identifiers seen at ingestion time.

    Deterministic and free -- no LLM call, which matters on a free tier. Zero
    false positives by construction: an unindexed token cannot match.
    """
    hits: set[str] = set()
    for raw in TOKEN_RE.findall(query):
        token = raw.strip(".-_")
        if token and token.lower() in vocab:
            hits.add(vocab[token.lower()])
    return sorted(hits)


def rrf(
    lanes: dict[str, tuple[list[str], float]],
) -> list[tuple[str, float, tuple[str, ...]]]:
    """Reciprocal Rank Fusion.

    Ranks, not scores: ts_rank_cd and cosine similarity live on incompatible
    scales, and normalising them means inventing constants. RRF needs only
    the ordering.
    """
    scores: dict[str, float] = {}
    seen_in: dict[str, list[str]] = {}
    for name, (ranked, weight) in lanes.items():
        for rank, uid in enumerate(ranked, start=1):
            scores[uid] = scores.get(uid, 0.0) + weight / (RRF_K + rank)
            seen_in.setdefault(uid, []).append(name)
    return sorted(
        ((uid, s, tuple(seen_in[uid])) for uid, s in scores.items()),
        key=lambda t: t[1],
        reverse=True,
    )


async def load_vocab(
    conn: asyncpg.Connection, corpus: str = DEFAULT_CORPUS
) -> dict[str, str]:
    rows = await conn.fetch(VOCAB_SQL, corpus)
    return {r["ident"].lower(): r["ident"] for r in rows}


async def search(
    conn: asyncpg.Connection,
    query: str,
    query_vector: str | None,
    vocab: dict[str, str],
    k: int = K_DEFAULT,
    corpus: str = DEFAULT_CORPUS,
) -> list[Hit]:
    """Run all lanes, fuse, hydrate the top k.

    `corpus` is not optional in spirit: several corpora share this table, and
    without the filter a TechQA question would compete against Chatwoot's
    deployment docs, silently contaminating both evals.
    """
    lex = [
        r["chunk_uid"] for r in await conn.fetch(LEXICAL_SQL, query, CANDIDATES, corpus)
    ]

    vec: list[str] = []
    if query_vector:
        vec = [
            r["chunk_uid"]
            for r in await conn.fetch(VECTOR_SQL, query_vector, CANDIDATES, corpus)
        ]

    idents = detect_identifiers(query, vocab)

    exact = (
        [
            r["chunk_uid"]
            for r in await conn.fetch(EXACT_SQL, idents, CANDIDATES, corpus)
        ]
        if idents
        else []
    )

    lanes = {"lexical": (lex, 1.0), "vector": (vec, 1.0)}
    if exact:
        lanes["exact"] = (exact, IDENT_BOOST)

    fused = rrf(lanes)[:k]
    if not fused:
        return []

    rows = {
        r["chunk_uid"]: r
        for r in await conn.fetch(HYDRATE_SQL, [uid for uid, _, _ in fused])
    }

    out = []
    for uid, score, in_lanes in fused:
        row = rows.get(uid)
        if row is None:
            continue
        out.append(
            Hit(
                chunk_uid=uid,
                score=score,
                lanes=in_lanes,
                doc_path=row["doc_path"],
                breadcrumb=row["breadcrumb"],
                content=row["content"],
                has_code=row["has_code"],
            )
        )
    return out
