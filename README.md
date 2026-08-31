# grounded

A support-deflection RAG agent that answers questions from a documentation
set, cites the exact passage it used, and — measurably — says "I don't know"
when the answer isn't there.

**[Live demo](#)** · **[90-second walkthrough](#)**

Built as a technical case study, not a toy: everything below is a number I
measured, not a number I expect. Where a design choice turned out to be
wrong, or a measurement changed the design, I've said so — that's the part
most write-ups leave out.

---

## The problem this solves

Off-the-shelf "chat with your docs" bots are commodity work — there are
hundreds of near-identical implementations. The differentiator isn't the
demo, it's whether the system's accuracy and failure modes are _measured_
rather than asserted. Almost none are. This one is, on two independent
corpora, with the eval code and raw results committed alongside the system
itself.

## Architecture

```
┌─────────────┐     ┌──────────────────────────┐     ┌────────────────┐
│   ingest    │────▶│  hybrid retrieval (RRF)   │────▶│  answer + cite │
│ (offline)   │     │  lexical + vector + exact │     │  or refuse     │
└─────────────┘     └──────────────────────────┘     └────────────────┘
 chunk, embed          Postgres + pgvector            Gate 1: retrieval score
 commit to git          no ANN index (~200 rows)      Gate 2: citation presence
                                                        multi-provider failover
```

- **FastAPI** backend, **Postgres + pgvector** for storage, **React-free
  vanilla JS** frontend (no build step — one HTML file)
- **LangGraph-style agent loop** without the framework: a linear
  retrieve → prompt → validate pipeline, because the control flow is simple
  enough that a graph library adds indirection without adding safety
- **Multi-provider LLM gateway** (Gemini + Groq) with automatic failover,
  disk caching, and client-side rate limiting
- Corpus and embeddings are **committed to the repo** — the deployed demo
  boots from `data/*.json` + `*.npy` with zero API calls, so a wiped
  database or an exhausted daily quota can't take it offline

## Why these design choices

**No ANN index.** At ~200 chunks, an exact sequential scan over the vector
column is single-digit milliseconds. HNSW/IVFFlat would add build time,
memory pressure, and approximate recall to solve a problem that doesn't
exist yet. Documented threshold to revisit: ~50k+ chunks.

**RRF fusion, not weighted score blending.** Cosine similarity and BM25's
`ts_rank_cd` live on incompatible scales; normalizing them means inventing
constants. Reciprocal Rank Fusion needs only the rank ordering from each
lane, which sidesteps the problem entirely.

**Two independent refusal gates**, because they catch different failures:

- _Gate 1 (retrieval confidence)_ — below a score threshold, no model call
  happens at all. Free, and catches the obviously out-of-scope question.
- _Gate 2 (citation presence)_ — the model must mark every claim with a
  numbered citation `[1]`, `[2]`. No valid citation, and the draft is
  discarded and replaced with a refusal. This is the one that matters: it
  catches retrieval returning _plausible_ sources and the model answering
  from its own pretraining instead of them — the failure mode that actually
  damages trust, because the answer looks right.

**Why two gates and not one.** Measured on TechQA (below): retrieval returns
a confident-looking top result for **100% of unanswerable questions**.
Vector search always returns its top-k regardless of similarity; lexical
matched something for all of them too. Retrieval _cannot_ refuse — the
citation gate is the only thing standing between the system and a
hallucination on every out-of-scope question it sees.

## Corpora

Two corpora serve two different jobs:

|         | Chatwoot docs                                          | TechQA-RAG-Eval                                                                                                                                |
| ------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| role    | **live flagship demo**                                 | **eval only**                                                                                                                                  |
| source  | [chatwoot/docs](https://github.com/chatwoot/docs), MIT | [nvidia/TechQA-RAG-Eval](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval), HF                                                           |
| content | deployment guides, API/config reference                | real IBM developer-forum support threads                                                                                                       |
| size    | 88 files → 187 chunks                                  | 184 docs → 266 chunks (300-question sample)                                                                                                    |
| why     | recognizable product, code-heavy, exact-match-friendly | **labeled unanswerable questions** + real support-style natural-language queries — things Chatwoot's docs can't provide without inventing them |

The Chatwoot corpus alone can't calibrate a refusal threshold or measure
natural-language recall honestly, because there's no ground truth for
"unanswerable" without writing it myself — and grading your own homework is
a bad look. TechQA ships 300 real forum questions, 33% labeled
`is_impossible`, which is exactly what a refusal eval needs.

## Retrieval: measured, not assumed

### Chatwoot — identifier-query recall@5 (n=296)

Auto-generated from 148 identifiers (env vars, config keys, CLI names) that
appear in exactly one chunk each — the gold chunk is known by construction,
so this needed no hand-labeling.

| lane                             | all       | bare (`REDIS_URL`) | natural ("what does X configure?") |
| -------------------------------- | --------- | ------------------ | ---------------------------------- |
| lexical (BM25)                   | 0.865     | 0.980              | 0.750                              |
| vector (gemini-embedding-001)    | 0.696     | 0.682              | 0.709                              |
| **hybrid (RRF, lexical+vector)** | **0.949** | **0.986**          | **0.912**                          |

**Finding:** vector search loses to plain keyword search on _every_ query
type on this corpus — 53% of chunks are code/config, which general-purpose
embedders represent poorly. But fusing it with lexical still lifts natural
-language recall from 0.750 to 0.912. The two lanes fail on different
queries; a component with a worse standalone average still adds real value
when its errors are uncorrelated with the other lane's.

_(`exact` and `fused-with-exact` lanes hit 1.000 by construction — gold is
defined as "the chunk containing identifier X," and the exact lane matches
directly on that same array. Reported in the raw results, excluded from the
comparison above as not an independent measurement.)_

### TechQA — pooled recall@5 (n=200 answerable, BGE-base-en-v1.5)

Real 44-word forum-post questions against all 184 documents pooled into one
index — genuine retrieval, not a single-document lookup.

| lane                         | recall@5  |
| ---------------------------- | --------- |
| lexical (BM25)               | 0.645     |
| **vector**                   | **0.960** |
| exact (identifiers)          | 0.020     |
| hybrid (RRF, lexical+vector) | 0.875     |

**Finding: the result inverts.** On Chatwoot, vector was the weak lane and
fusion won. Here vector dominates and **fusion actively hurts** — equal-
weight RRF drags a strong lane down toward a weak one. Prose support
articles reward semantic similarity; code-and-config docs reward term
matching. There's no universal answer — the two corpora prove it in opposite
directions with the same code.

_(Caveat: the two corpora use different embedding models — Gemini for
Chatwoot, local BGE for TechQA, to stay under Gemini's free daily quota —
so part of the gap could be model, not corpus. Re-embedding Chatwoot with
BGE to isolate the variable is a documented next step, not yet done.)_

## Answer layer: does it actually refuse?

Measured on TechQA (n=60: 40 answerable, 20 impossible, k=3 retrieved
chunks), three configurations:

|                                                | gpt-oss-120b | gpt-oss-120b (wider ctx) | gemini-3.5-flash-lite |
| ---------------------------------------------- | ------------ | ------------------------ | --------------------- |
| refusal rate (of 20 impossible)                | 0.900        | **0.950**                | 0.700                 |
| false answers shipped                          | 2/20         | **1/20**                 | 6/20                  |
| answer rate (of 40 answerable)                 | 0.425        | **0.675**                | 0.475                 |
| citation accuracy                              | **0.941**    | 0.852                    | 0.941¹                |
| over-refusal (declined despite gold retrieved) | 0.500        | **0.219**                | 0.406                 |
| latency (p50)                                  | 14.9s        | 20.2s                    | **9.9s**              |

¹ citation accuracy is high partly _because_ flash-lite only answers the
easy questions — the ones where the top-ranked chunk is obviously right.

**Findings, in the order I found them:**

1. **Citation format is not cosmetic.** The first prompt asked the model to
   cite by 16-character hex chunk ID. Models copy long hex strings
   incorrectly — truncated, one character off, occasionally invented
   outright — and every slip was scored as "no valid citation" even when
   the answer was correctly grounded. Switching to small numeric labels
   (`[1]`, `[2]`) that map back to chunk IDs server-side cut that failure
   mode roughly 3x (9 → 3 refusals from this cause alone) with no other
   change.
2. **Reasoning-model overhead buys refusal discipline, at a real latency
   cost.** `gpt-oss-120b` beats `flash-lite` on refusal rate, false-answer
   rate, and answer rate — the numbers that determine whether a client's
   users get correct information — at roughly 2x the latency. For a
   support bot, correctness dominates; for a live chat demo, 20 seconds is
   at the edge of what a visitor will wait for. Both are true, and the
   trade-off is the point: there's no single "best" config, only a curve
   the client should be shown.
3. **Retrieval cannot refuse (see false-hit rate below).** The citation
   gate is not a nice-to-have on top of good retrieval — on this corpus,
   it is the _only_ thing preventing a confident wrong answer on every
   single unanswerable question retrieval sees.

### False-hit rate on unanswerable questions (n=100, TechQA, retrieval only)

| lane    | returns _something_ for a question with no correct answer |
| ------- | --------------------------------------------------------- |
| lexical | 1.000                                                     |
| vector  | 1.000                                                     |
| hybrid  | 1.000                                                     |
| exact   | 0.000 (silent when no identifier is present — expected)   |

Every retrieval lane except the identifier-exact-match lane hands the
generator a plausible-looking source for a question it cannot answer, 100%
of the time. This is _why_ the answer layer has two gates instead of one,
and it's a claim I can point at a number for instead of asserting.

## Cost and latency, on free-tier infrastructure

- ~2,600 tokens/question, ~$0 marginal cost on free-tier quota
- Groq (chat) and Gemini (chat + embeddings) both cap around
  1,000 requests/day on the free tier — enough for real demo traffic, not
  enough to iterate on a 300-question eval in one sitting. Evals in this
  repo use local embeddings (BGE-base-en-v1.5, sentence-transformers) to
  avoid burning that quota on retrieval measurement, keeping it for the
  chat-generation numbers that actually need a hosted model.
- Both LLM providers are called through a **failover gateway**: a 429/5xx/
  network error on one provider retries with backoff, then fails to the
  next provider, then degrades to "sources found, generation unavailable"
  rather than a 500. Verified against a real deprecated-model 404 and a
  real daily-cap 429 during development — see [Things that broke](#things-that-broke-and-what-that-says).
- No ANN index, no Redis (in-process rate limiter, single instance),
  corpus + embeddings committed to git — the whole system is designed to
  survive being unattended on a free tier for months, not just to demo
  well once.

## Things that broke, and what that says

Kept deliberately, because a portfolio piece that only shows the happy path
teaches a reader nothing about how the author handles the rest of it:

- **`plainto_tsquery` ANDs every search term.** Fine for a 2-word config-key
  query, silently returns zero results for a 44-word real support question.
  Measured as recall@5 = 0.000 on first TechQA run. Fixed by ORing the
  lexemes and letting `ts_rank_cd` do the ranking it's designed for.
- **A colon in a token is tsquery syntax.** A support question containing a
  pasted stack trace (`SocketInputStream.java:140`) crashed the entire eval
  run with a SQL syntax error. Real support users paste stack traces
  constantly; an unquoted lexeme in production would 500 on that input.
  Fixed by quoting every lexeme.
- **An unnamespaced LLM response cache silently served stub answers to a
  real eval run.** The report looked clean — refusal rate, citation
  accuracy, full table — and was entirely fictitious, because the cache key
  didn't include the model name. The only tell was a "0 live API calls"
  counter, which is why that counter is now a permanent line in every eval
  report.
- **Provider "failover" only handled rate limits, not deprecated models or
  oversized requests.** A hardcoded chat-model name went stale (both
  Gemini's and Groq's model catalogs changed during development) and threw
  404s that killed the run instead of failing over — model deprecation is
  a more likely outage than a rate limit on a demo that stays live for
  months. Fixed by widening the retryable-status set and moving model
  names to environment variables instead of hardcoding them.

## Stack

FastAPI · PostgreSQL + pgvector · LangChain-free agent loop · Gemini +
Groq (provider-agnostic gateway) · BGE-base-en-v1.5 (local eval embeddings)
· vanilla JS frontend, no build step · Docker Compose (local Postgres only)
· `uv` for dependency management

## Running it locally

```bash
docker compose up -d                      # postgres + pgvector
uv run --group ingest scripts/run_ingest.py
uv run scripts/load_chunks.py
uv run --group ingest scripts/embed_chunks.py
uv run uvicorn src.api.main:app --reload  # API on :8000
python -m http.server 5500 --directory web  # frontend on :5500
```

Reproduce the numbers above:

```bash
uv run evals/run_retrieval.py                 # Chatwoot identifier eval
uv run evals/compare_lanes.py --corpus techqa  # TechQA lane comparison
uv run evals/run_answers.py --limit 60         # answer-layer eval
```

Raw results for every table above are committed under `evals/results/`.

## Honest limitations

- Chatwoot and TechQA use different embedding models, so vector-recall
  numbers aren't directly comparable across corpora — noted above, not
  hidden.
- The answer-layer eval ran at n=60, not the full 300-question set, due to
  free-tier daily token caps. Directionally reliable, not a large-sample
  guarantee.
- MIN_SCORE (Gate 1's retrieval-confidence threshold) is set from a rough
  sweep against TechQA's labeled impossible questions; it hasn't been
  re-validated against Chatwoot's own out-of-scope traffic.
- No production traffic yet — every number above is from an offline eval,
  not live users.

## Attribution

Chatwoot documentation vendored from
[chatwoot/docs](https://github.com/chatwoot/docs) (MIT License) at commit
`376277a46614bf9b1fb59c173212f56fcac07ae0`. TechQA-RAG-Eval from
[nvidia/TechQA-RAG-Eval](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval)
on Hugging Face.
