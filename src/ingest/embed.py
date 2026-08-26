"""Embedding with an on-disk cache and a client-side rate limiter.

Three things matter under a free-tier quota:

1. **Never pay for the same text twice.** Every embedding is cached by
   sha256(model|task_type|text). Re-running the eval 20 times costs one round
   of API calls, not twenty.

2. **Pace requests, don't just retry them.** Free tier for Gemini Embedding is
   roughly 100 RPM / 30k TPM / 1000 RPD. A burst of 187 requests blows the
   per-minute window instantly, and exponential backoff after the fact just
   wastes wall-clock time. A token bucket keeps us under the limit in the
   first place; backoff is the fallback, not the strategy.

   Note also that batchEmbedContents carries its OWN much smaller per-region
   quota than embedContent, which is why this uses single calls.

3. **Asymmetric task types.** Documents embed with RETRIEVAL_DOCUMENT and
   queries with RETRIEVAL_QUERY. Gemini trains these as distinct objectives;
   using one type for both is a quiet accuracy loss.

The API key travels in the x-goog-api-key header, never a query parameter --
query strings leak into tracebacks, proxy logs and shell history.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

DIM = 768
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / ".embed_cache"


class Embedder(Protocol):
    model: str

    async def embed(self, texts: list[str], task_type: str) -> list[list[float]]: ...


def _key(model: str, task_type: str, text: str) -> str:
    return hashlib.sha256(f"{model}|{task_type}|{text}".encode()).hexdigest()


class Cache:
    def __init__(self, directory: Path = CACHE_DIR) -> None:
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> list[float] | None:
        path = self.dir / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, key: str, vector: list[float]) -> None:
        (self.dir / f"{key}.json").write_text(json.dumps(vector))


class RateLimiter:
    """Simple token bucket. Keeps us under RPM instead of discovering it."""

    def __init__(self, per_minute: int) -> None:
        self.interval = 60.0 / per_minute
        self._next = 0.0

    async def wait(self) -> None:
        now = time.monotonic()
        if now < self._next:
            await asyncio.sleep(self._next - now)
        self._next = max(now, self._next) + self.interval


class GeminiEmbedder:
    model = "gemini-embedding-001"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"

    def __init__(self, api_key: str | None = None, rpm: int | None = None) -> None:
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        # Deliberately below the documented 100 RPM: TPM (~30k) usually binds
        # first on chunks this size, and headroom is cheaper than a stall.
        self.limiter = RateLimiter(rpm or int(os.getenv("EMBED_RPM", "60")))

    async def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        headers = {"x-goog-api-key": self.api_key}
        async with httpx.AsyncClient(timeout=60, headers=headers) as client:
            for i, text in enumerate(texts, start=1):
                await self.limiter.wait()
                out.append(await self._one(client, text, task_type))
                if i % 25 == 0 or i == len(texts):
                    print(f"    {i}/{len(texts)}")
        return out

    async def _one(
        self, client: httpx.AsyncClient, text: str, task_type: str, attempt: int = 0
    ) -> list[float]:
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": DIM,
        }
        resp = await client.post(self.URL.format(model=self.model), json=payload)
        if resp.status_code == 429 and attempt < 6:
            wait = float(resp.headers.get("retry-after", 0)) or 2**attempt * 5
            print(f"    429 -- waiting {wait:.0f}s (attempt {attempt + 1})")
            await asyncio.sleep(wait)
            return await self._one(client, text, task_type, attempt + 1)
        if resp.status_code == 429:
            raise RuntimeError(
                "Persistent 429. Likely the daily cap (RPD) rather than RPM -- "
                "it resets at midnight Pacific. Cached vectors are kept, so "
                "re-running resumes where this stopped."
            )
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]


class HashEmbedder:
    """Deterministic offline stand-in. NOT semantic -- plumbing tests only."""

    model = "hash-dev"

    async def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        out = []
        for text in texts:
            seed = int(
                hashlib.sha256(f"{task_type}|{text}".encode()).hexdigest()[:16], 16
            )
            v = np.random.default_rng(seed).normal(size=DIM)
            out.append((v / np.linalg.norm(v)).tolist())
        return out


async def embed_cached(
    embedder: Embedder,
    texts: list[str],
    task_type: str,
    cache: Cache | None = None,
    label: str = "",
) -> np.ndarray:
    """Embed texts, hitting the API only for cache misses.

    Fresh vectors are written to the cache as they arrive, so an interrupted
    run loses nothing -- important when a daily quota can cut you off midway.
    """
    cache = cache or Cache()
    keys = [_key(embedder.model, task_type, t) for t in texts]
    vectors: list[list[float] | None] = [cache.get(k) for k in keys]

    missing = [i for i, v in enumerate(vectors) if v is None]
    if not missing:
        print(f"{label}: all {len(texts)} cached")
        return np.array(vectors, dtype=np.float32)

    print(f"{label}: {len(missing)} to embed, {len(texts) - len(missing)} cached")
    fresh = await embedder.embed([texts[i] for i in missing], task_type)
    for i, vector in zip(missing, fresh):
        cache.put(keys[i], vector)
        vectors[i] = vector
    return np.array(vectors, dtype=np.float32)


def get_embedder() -> Embedder:
    if os.getenv("EMBED_PROVIDER", "gemini") == "hash":
        return HashEmbedder()
    return GeminiEmbedder()
