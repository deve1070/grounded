#!/usr/bin/env python
"""Measure the answer layer: does Gate 2 catch what retrieval cannot refuse?

    uv run evals/run_answers.py [--limit 60] [--corpus techqa]

compare_lanes.py showed a false-hit rate of 1.000: all 100 unanswerable
questions get confident-looking sources. Retrieval structurally cannot refuse.
So the citation gate is the only thing between the system and 100 confident
wrong answers. This measures whether it works.

Four numbers, all from labels that already exist -- no LLM judge, so nothing
here depends on a second model's opinion:

  refusal rate      impossible questions correctly declined. THE headline.
  false answer rate impossible questions answered anyway. Every one of these
                    is a hallucination shipped to a user.
  answer rate       answerable questions actually answered (over-refusal is a
                    real cost too -- a bot that always declines is useless)
  citation accuracy answered questions whose citations include a gold chunk.
                    Catches answers that are right-looking but sourced from
                    the wrong place.

Start with --limit 60. It costs 60 calls and tells you whether the prompt
works before you spend a full run on it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.agent.answer import answer
from src.ingest.embed import embed_cached, get_embedder
from src.llm.gateway import Gateway
from src.retrieval.search import load_vocab, search

QUESTIONS = ROOT / "evals" / "answers" / "techqa_questions.json"
RESULTS = ROOT / "evals" / "results" / "answer_eval.json"
DSN = os.environ["DATABASE_URL"]
# Fewer, fuller chunks beat more truncated ones -- and on a token-limited free
# tier, k is the cheapest lever on cost per question.
K = int(os.getenv("RETRIEVAL_K", "3"))


def balanced_sample(questions: list[dict], limit: int) -> list[dict]:
    """Keep the answerable/impossible ratio when sampling."""
    imp = [q for q in questions if q["is_impossible"]][: limit // 3]
    ans = [q for q in questions if not q["is_impossible"]][: limit - len(imp)]
    return ans + imp


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--corpus", default="techqa")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS.read_text())
    if args.limit:
        questions = balanced_sample(questions, args.limit)

    gateway = Gateway()
    if not gateway.providers:
        print(
            "no LLM providers configured -- set GROQ_API_KEY or "
            "GEMINI_API_KEY in .env, or LLM_PROVIDER=echo to smoke-test"
        )
        return 1
    print(f"providers: {[(p.name, p.model) for p in gateway.providers]}")
    print(
        f"questions: {len(questions)} "
        f"({sum(not q['is_impossible'] for q in questions)} answerable, "
        f"{sum(q['is_impossible'] for q in questions)} impossible)\n"
    )

    conn = await asyncpg.connect(DSN)
    records = []
    try:
        vocab = await load_vocab(conn, args.corpus)
        arr = await embed_cached(
            get_embedder(),
            [q["query"] for q in questions],
            task_type="RETRIEVAL_QUERY",
            label="queries",
        )
        qvec = {
            q["query"]: "[" + ",".join(f"{x:.6f}" for x in v) + "]"
            for q, v in zip(questions, arr)
        }

        for i, q in enumerate(questions, start=1):
            hits = await search(
                conn, q["query"], qvec[q["query"]], vocab, k=K, corpus=args.corpus
            )
            started = time.perf_counter()
            result = await answer(q["query"], hits, gateway)
            elapsed = int((time.perf_counter() - started) * 1000)

            gold = set(q["gold_chunk_uids"])
            records.append(
                {
                    "id": q["id"],
                    "impossible": q["is_impossible"],
                    "refused": result.refused,
                    "error": result.error,
                    "reason": result.reason,
                    "citations": result.citations,
                    "cited_gold": bool(gold & set(result.citations)),
                    "gold_retrieved": bool(gold & {h.chunk_uid for h in hits}),
                    "latency_ms": elapsed,
                    # Recorded per row, not assumed from the environment: an env
                    # var that silently fails to take is exactly how three runs
                    # got attributed to the wrong model.
                    "provider": result.provider,
                    "model": result.model,
                    "cached": result.cached,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "text": result.text[:400],
                }
            )
            if i % 10 == 0 or i == len(questions):
                print(f"  {i}/{len(questions)}")
    finally:
        await conn.close()

    # Provider failures are NOT refusals. A rate-limited run that scored
    # every 429 as "the system declined" would report a perfect refusal rate
    # while having answered almost nothing -- which is exactly what happened
    # on the first attempt at this eval.
    errors = [r for r in records if r["error"]]
    scored = [r for r in records if not r["error"]]
    imp = [r for r in scored if r["impossible"]]
    ans = [r for r in scored if not r["impossible"]]

    if errors:
        print(
            f"\n!! {len(errors)}/{len(records)} questions failed at the "
            f"provider and are EXCLUDED from every number below."
        )
        for reason, n in Counter(r["reason"] for r in errors).most_common(3):
            print(f"   {reason}  x{n}")
        if len(errors) > len(records) * 0.2:
            print(
                "   More than a fifth of the run failed -- lower LLM_RPM or "
                "add a second provider before trusting these results."
            )
    if not imp or not ans:
        print("\nnot enough successful calls to score. Re-run when quota allows.")
        return 1

    refused_imp = sum(r["refused"] for r in imp)
    answered_ans = sum(not r["refused"] for r in ans)
    cited_gold = sum(r["cited_gold"] for r in ans if not r["refused"])
    # Over-refusal only counts where retrieval DID find the gold chunk --
    # otherwise refusing was the correct call and the fault is retrieval's.
    recoverable = [r for r in ans if r["gold_retrieved"]]
    missed = sum(r["refused"] for r in recoverable)

    print("\n" + "=" * 58)
    print(
        f"IMPOSSIBLE QUESTIONS ({len(imp)}) -- retrieval gave all of these"
        "\nconfident sources; only the citation gate can stop them"
    )
    print("=" * 58)
    print(
        f"refusal rate        {refused_imp / len(imp):>7.3f}  "
        f"({refused_imp}/{len(imp)})"
    )
    print(
        f"FALSE ANSWER rate   {(len(imp) - refused_imp) / len(imp):>7.3f}  "
        f"({len(imp) - refused_imp}/{len(imp)})  <- hallucinations shipped"
    )

    print("\n" + "=" * 58)
    print(f"ANSWERABLE QUESTIONS ({len(ans)})")
    print("=" * 58)
    print(
        f"answer rate         {answered_ans / len(ans):>7.3f}  "
        f"({answered_ans}/{len(ans)})"
    )
    if answered_ans:
        print(
            f"citation accuracy   {cited_gold / answered_ans:>7.3f}  "
            f"({cited_gold}/{answered_ans} cite a gold chunk)"
        )
    if recoverable:
        print(
            f"over-refusal        {missed / len(recoverable):>7.3f}  "
            f"({missed}/{len(recoverable)} refused despite gold retrieved)"
        )

    print("\n" + "=" * 58)
    print("REFUSAL REASONS")
    print("=" * 58)
    for reason, n in Counter(r["reason"] for r in scored if r["refused"]).most_common():
        print(f"  {reason:26} {n}")

    live = [r for r in scored if not r["cached"]]
    lat = sorted(r["latency_ms"] for r in live) or [0]
    tokens = sum(r["input_tokens"] + r["output_tokens"] for r in scored)
    print("\n" + "=" * 58)
    print("COST AND LATENCY")
    print("=" * 58)
    print(
        f"scored              {len(scored)}/{len(records)} "
        f"({len(errors)} provider failures excluded)"
    )
    print(
        f"live calls          {len(live)}/{len(scored)} "
        f"({len(scored) - len(live)} served from cache)"
    )
    print(f"latency p50/p95     {lat[len(lat) // 2]} / {lat[int(len(lat) * 0.95)]} ms")
    print(
        f"tokens total        {tokens}  "
        f"(~{tokens / max(len(scored), 1):.0f} per question)"
    )
    used = sorted({r["model"] for r in scored if r["model"]})
    print(f"model(s) that answered  {used or ['none - all refused pre-call']}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "corpus": args.corpus,
                "k": K,
                "n_attempted": len(records),
                "n_scored": len(scored),
                "n_provider_errors": len(errors),
                "providers": [
                    {"name": p.name, "model": p.model} for p in gateway.providers
                ],
                "models_used": sorted({r["model"] for r in records if r["model"]}),
                "embedding_model": get_embedder().model,
                "refusal_rate": round(refused_imp / len(imp), 3) if imp else None,
                "false_answer_rate": round((len(imp) - refused_imp) / len(imp), 3)
                if imp
                else None,
                "answer_rate": round(answered_ans / len(ans), 3) if ans else None,
                "citation_accuracy": round(cited_gold / answered_ans, 3)
                if answered_ans
                else None,
                "latency_p50_ms": lat[len(lat) // 2],
                "records": records,
            },
            indent=2,
        )
    )
    print(f"\n-> {RESULTS}")
    print(
        "\nRead the false answers in the JSON. Each one is a prompt bug or a"
        "\nmislabelled question, and you cannot tell which from the numbers."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
