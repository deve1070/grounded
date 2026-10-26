from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / ".llm_cache"


class AllProvidersExhausted(RuntimeError):
    """Infrastructure failure, NOT a refusal. Callers must keep these apart:
    counting a rate-limited request as 'the system declined to answer' turns
    a broken eval run into a flattering refusal rate."""


class RateLimiter:
    """Token bucket. Free chat tiers are a few requests per minute, so pacing
    beats discovering the limit and backing off after the fact."""

    def __init__(self, per_minute: int) -> None:
        self.interval = 60.0 / max(per_minute, 1)
        self._next = 0.0

    async def wait(self) -> None:
        now = time.monotonic()
        if now < self._next:
            await asyncio.sleep(self._next - now)
        self._next = max(now, self._next) + self.interval


@dataclass
class Completion:
    text: str
    provider: str
    model: str
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


class GeminiChat:
    name = "gemini"

    @property
    def model(self) -> str:
        return os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")

    URL = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/{model}:generateContent"
    )

    def __init__(self) -> None:
        self.api_key = os.environ["GEMINI_API_KEY"]

    async def complete(
        self, system: str, user: str, max_tokens: int = 800
    ) -> Completion:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.1},
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.URL.format(model=self.model),
                headers={"x-goog-api-key": self.api_key},
                json=payload,
            )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(
            p.get("text", "") for p in data["candidates"][0]["content"]["parts"]
        )
        usage = data.get("usageMetadata", {})
        return Completion(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
        )


class GroqChat:
    name = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def model(self) -> str:
        return os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")

    def __init__(self) -> None:
        self.api_key = os.environ["GROQ_API_KEY"]

    async def complete(
        self, system: str, user: str, max_tokens: int = 800
    ) -> Completion:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return Completion(
            text=data["choices"][0]["message"]["content"],
            provider=self.name,
            model=self.model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


class EchoChat:
    """Offline stand-in so the agent and API can be exercised without quota."""

    name = "echo"
    model = "echo-dev"

    async def complete(
        self, system: str, user: str, max_tokens: int = 800
    ) -> Completion:
        first = next(
            (line for line in user.splitlines() if line.startswith("[")), "[1]"
        )
        uid = first.split("]")[0].lstrip("[")
        return Completion(
            text=f"Stub answer. [{uid}]", provider=self.name, model=self.model
        )


class Gateway:
    def __init__(
        self,
        providers: list | None = None,
        cache_dir: Path = CACHE_DIR,
        rpm: int | None = None,
    ) -> None:
        self.providers = providers if providers is not None else _default()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(rpm or int(os.getenv("LLM_RPM", "12")))

    def _key(self, model: str, system: str, user: str) -> str:
        """Cache key includes the model.

        Keying on (system, user) alone meant a stub run and a real run shared
        entries: EchoChat's canned answers were served to a live eval, which
        reported 0 live calls and completely fictitious numbers. Any cache
        that can serve one model's output for another model's request is a
        correctness bug, not an optimisation.
        """
        return hashlib.sha256(f"{model}\x00{system}\x00{user}".encode()).hexdigest()

    MAX_RETRY_WAIT = 60

    async def _with_retry(
        self, provider, system: str, user: str, max_tokens: int, attempts: int = 3
    ) -> Completion:
        """Retry 429 on THIS provider before failing over.

        Failing over on the first 429 burns the second provider's quota to
        work around a limit that clears in seconds.
        """
        for attempt in range(attempts):
            await self.limiter.wait()
            try:
                return await provider.complete(system, user, max_tokens)
            except httpx.RequestError as exc:
                if attempt == attempts - 1:
                    raise
                print(
                    f"  {provider.name} network error ({type(exc).__name__})"
                    f" -- retrying"
                )
                await asyncio.sleep(2**attempt * 2)
            except httpx.HTTPStatusError as exc:
                retryable = exc.response.status_code in (429, 500, 502, 503, 529)
                if not retryable or attempt == attempts - 1:
                    raise
                wait = (
                    float(exc.response.headers.get("retry-after", 0)) or 2**attempt * 5
                )
                if wait > self.MAX_RETRY_WAIT:
                    print(
                        f"  {provider.name} rate limited for {wait:.0f}s "
                        f"(daily cap) -- failing over instead of waiting"
                    )
                    raise
                print(
                    f"  {provider.name} {exc.response.status_code} -- "
                    f"waiting {wait:.0f}s"
                )
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable")

    async def complete(
        self, system: str, user: str, max_tokens: int = 800
    ) -> Completion:
        errors = []
        for provider in self.providers:
            path = self.cache_dir / f"{self._key(provider.model, system, user)}.json"
            if path.exists():
                data = json.loads(path.read_text())
                return Completion(**{**data, "cached": True})

            try:
                result = await self._with_retry(provider, system, user, max_tokens)
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code

                if code in (400, 401, 403, 404, 413, 429, 500, 502, 503, 529):
                    if code not in (429, 500, 502, 503, 529):
                        print(
                            f"  {provider.name} unavailable ({code}) -- "
                            f"check model name and key; failing over"
                        )
                    errors.append(f"{provider.name}:{code}")
                    continue
                raise
            except httpx.RequestError as exc:
                errors.append(f"{provider.name}:network({type(exc).__name__})")
                continue
            except KeyError as exc:  # missing API key for this provider
                errors.append(f"{provider.name}:no-key({exc})")
                continue

            path.write_text(
                json.dumps(
                    {
                        "text": result.text,
                        "provider": result.provider,
                        "model": result.model,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    }
                )
            )
            return result

        raise AllProvidersExhausted("; ".join(errors) or "no providers")


def _default() -> list:
    if os.getenv("LLM_PROVIDER") == "echo":
        return [EchoChat()]
    providers: list = []
    if os.getenv("GEMINI_API_KEY"):
        providers.append(GeminiChat())
    if os.getenv("GROQ_API_KEY"):
        providers.append(GroqChat())
    return providers
