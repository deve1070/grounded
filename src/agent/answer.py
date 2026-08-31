"""The answer contract: cite or refuse.

Two independent gates, because they catch different failures.

  Gate 1 -- retrieval confidence. If the best fused score is below
  MIN_SCORE and no identifier was detected, we do not call the model at all.
  Cheap (no tokens spent) and catches the clearly out-of-scope question.

  Gate 2 -- citation presence. The model is instructed to mark every claim
  with [chunk_uid]. If the reply contains no valid citation, we discard it and
  refuse. This catches the dangerous case that Gate 1 cannot: retrieval
  returned plausible-looking chunks, and the model answered from its own
  pretraining instead of from them.

Refusal is a feature, not a limitation. "I don't know, here is how to reach
support" is what makes the rest of the answers trustworthy -- and it is the
behaviour a buyer is actually screening for.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from src.llm.gateway import AllProvidersExhausted, Completion, Gateway
from src.retrieval.search import Hit

MIN_SCORE = 0.028  # ~ 1/(60+2): weak unless a lane ranked it near top

# Prompt size has to be bounded. Five chunks of up to 800 words each is a
# ~25 KB request, which Groq's free tier rejects outright with 413. Even where
# it is accepted, an unbounded prompt means unpredictable latency and cost per
# question -- both numbers a client asks about. Budget in characters rather
# than tokens: no tokenizer dependency, and erring small is cheap.
# Tokens-per-minute is usually the binding free-tier limit, not requests: a
# 12k-char prompt is ~3k tokens, so even 8 requests/minute can exceed a 10k
# TPM allowance. Shrinking the prompt buys more throughput than slowing down,
# and it cuts cost per question at the same time.
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
MIN_CHUNK_CHARS = 400  # below this a chunk is too clipped to cite honestly
# Sources are labelled [1], [2], [3] rather than by chunk_uid. A 16-character
# hex id is a string models copy wrong -- they truncate it, alter a character,
# or invent a plausible-looking one, and every such slip is scored as a
# refusal even though the answer was correctly grounded. Small integers are
# copied reliably, and the label maps back to a uid on our side.
CITATION_RE = re.compile(r"\[(\d{1,2})\]")

REFUSAL_TEXT = (
    "I don't have that in the documentation I've been given. "
    "You may want to contact support directly."
)

SYSTEM = """You answer questions using ONLY the provided sources.

Rules:
- Every factual claim must end with a citation in the form [1], [2] etc, \
using the number shown in the source header.
- If the sources do not contain the answer, reply with exactly: NO_ANSWER
- Do not use knowledge from outside the sources, even if you are confident.
- Prefer quoting exact configuration keys, commands and values verbatim.
- Be concise. Two or three sentences unless steps are required."""


@dataclass
class Answer:
    text: str
    refused: bool
    # True when no answer was produced because the providers failed, not
    # because the system judged the sources insufficient. Evals must exclude
    # these rather than score them as refusals.
    error: bool = False
    reason: str = ""
    citations: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


def build_prompt(query: str, hits: list[Hit]) -> tuple[str, list[Hit]]:
    """Assemble the prompt within the character budget.

    Returns the prompt AND the hits that actually made it in. Citations are
    validated against the latter: a chunk that was dropped for budget was
    never shown to the model, so a citation naming it cannot be genuine.

    Hits arrive in rank order, so the budget is spent on the best candidates
    first. A chunk that only partly fits is truncated rather than dropped,
    since the top of a support article usually carries the answer -- but only
    if enough survives to cite honestly.
    """
    blocks: list[str] = []
    used: list[Hit] = []
    budget = MAX_CONTEXT_CHARS

    for n, hit in enumerate(hits, start=1):
        header = f"[{n}] {hit.breadcrumb}\n"
        available = budget - len(header) - 20
        if available < MIN_CHUNK_CHARS:
            break
        body = hit.content
        if len(body) > available:
            body = body[:available].rsplit(" ", 1)[0] + "\n[... truncated]"
        blocks.append(header + body)
        used.append(hit)
        budget -= len(header) + len(body) + 20

    prompt = "SOURCES\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQUESTION\n{query}"
    return prompt, used


def _refuse(reason: str, hits: list[Hit], error: bool = False) -> Answer:
    return Answer(
        text=REFUSAL_TEXT, refused=True, reason=reason, hits=hits, error=error
    )


async def answer(
    query: str,
    hits: list[Hit],
    gateway: Gateway,
    identifiers_found: bool = False,
    min_score: float = MIN_SCORE,
) -> Answer:
    if not hits:
        return _refuse("no_results", hits)

    # Gate 1: retrieval confidence, before spending any tokens.
    if hits[0].score < min_score and not identifiers_found:
        return _refuse("low_retrieval_score", hits)

    prompt, used = build_prompt(query, hits)
    try:
        completion: Completion = await gateway.complete(SYSTEM, prompt)
    except AllProvidersExhausted as exc:
        return _refuse(f"providers_exhausted:{exc}", hits, error=True)

    text = completion.text.strip()
    # Map the numeric labels the model used back to chunk uids, ignoring any
    # number outside the range we actually showed it.
    cited = []
    for label in CITATION_RE.findall(text):
        index = int(label) - 1
        if 0 <= index < len(used):
            cited.append(used[index].chunk_uid)

    # Gate 2: the model must ground its answer in the sources we gave it.
    if text.upper().startswith("NO_ANSWER") or not cited:
        reason = (
            "model_declined"
            if text.upper().startswith("NO_ANSWER")
            else "no_valid_citation"
        )
        result = _refuse(reason, hits)
        result.provider = completion.provider
        result.model = completion.model
        result.cached = completion.cached
        result.input_tokens = completion.input_tokens
        result.output_tokens = completion.output_tokens
        return result

    return Answer(
        text=text,
        refused=False,
        citations=sorted(set(cited)),
        hits=hits,
        provider=completion.provider,
        model=completion.model,
        cached=completion.cached,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )
