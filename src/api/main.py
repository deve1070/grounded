from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from src.agent.answer import answer
from src.api.limits import Limiter
from src.ingest.embed import embed_cached, get_embedder
from src.llm.gateway import Gateway
from src.retrieval.search import load_vocab, search

DSN = os.environ["DATABASE_URL"]
DEMO_CORPUS = os.getenv("DEMO_CORPUS", "chatwoot")
state: dict = {}
limiter = Limiter()


async def seed_if_empty(conn: asyncpg.Connection) -> None:
    """Load committed chunks + vectors when the table is empty."""
    if await conn.fetchval(
        "SELECT count(*) FROM chunks WHERE corpus = $1", DEMO_CORPUS
    ):
        return
    chunks = json.loads((ROOT / "data" / "chunks.json").read_text())
    vectors = np.load(ROOT / "data" / "embeddings.npy")
    uids = json.loads((ROOT / "data" / "embeddings_uids.json").read_text())
    by_uid = {u: v for u, v in zip(uids, vectors)}

    await conn.executemany(
        "INSERT INTO chunks (chunk_uid, corpus, doc_path, breadcrumb, "
        "heading_level, content, word_count, has_code, identifiers, embedding) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::vector) "
        "ON CONFLICT (chunk_uid) DO NOTHING",
        [
            (
                c["chunk_uid"],
                c["corpus"],
                c["doc_path"],
                c["breadcrumb"],
                c["heading_level"],
                c["content"],
                c["word_count"],
                c["has_code"],
                c["identifiers"],
                "[" + ",".join(f"{x:.6f}" for x in by_uid[c["chunk_uid"]]) + "]",
            )
            for c in chunks
            if c["chunk_uid"] in by_uid
        ],
    )
    print(f"seeded {len(chunks)} chunks from committed data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=3)
    async with pool.acquire() as conn:
        await seed_if_empty(conn)
        state["vocab"] = await load_vocab(conn, DEMO_CORPUS)
    state["pool"] = pool
    state["gateway"] = Gateway()
    yield
    await pool.close()


app = FastAPI(title="grounded", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )


@app.get("/health")
async def health() -> dict:
    async with state["pool"].acquire() as conn:
        n = await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE corpus = $1", DEMO_CORPUS
        )
    return {"status": "ok", "corpus": DEMO_CORPUS, "chunks": n, **limiter.stats()}


@app.post("/ask")
async def ask(body: Ask, request: Request) -> JSONResponse:
    ip = client_ip(request)
    started = time.perf_counter()

    decision = limiter.check(ip)
    if not decision.allowed:
        return JSONResponse(
            status_code=429,
            content={
                "refused": True,
                "reason": decision.reason,
                "retry_after_seconds": decision.retry_after,
                "message": (
                    "This demo runs on free-tier quota and has hit its "
                    "limit. The recorded walkthrough shows the same flow."
                ),
            },
        )

    # Query embedding is the one unavoidable runtime API call. If it fails,
    # drop to lexical + exact rather than failing the request: a degraded
    # answer with real citations beats an error page.
    query_vector = None
    try:
        arr = await embed_cached(
            get_embedder(), [body.question], task_type="RETRIEVAL_QUERY", label="q"
        )
        query_vector = "[" + ",".join(f"{x:.6f}" for x in arr[0]) + "]"
    except Exception as exc:  # noqa: BLE001
        print(f"embedding unavailable, lexical only: {exc}")

    async with state["pool"].acquire() as conn:
        hits = await search(
            conn, body.question, query_vector, state["vocab"], k=5, corpus=DEMO_CORPUS
        )

    limiter.record(ip)
    result = await answer(body.question, hits, state["gateway"])
    elapsed = int((time.perf_counter() - started) * 1000)

    return JSONResponse(
        content={
            "answer": result.text,
            "refused": result.refused,
            "reason": result.reason,
            "citations": [
                {
                    "chunk_uid": h.chunk_uid,
                    "breadcrumb": h.breadcrumb,
                    "doc_path": h.doc_path,
                    "cited": h.chunk_uid in result.citations,
                    "excerpt": h.content[:1800],
                }
                for h in result.hits
            ],
            "meta": {
                "latency_ms": elapsed,
                "provider": result.provider,
                "model": result.model,
                "cached": result.cached,
                "vector_lane": query_vector is not None,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
        }
    )
