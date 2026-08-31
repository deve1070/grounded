#!/usr/bin/env python
"""Second pass on TechQA-RAG-Eval: decide the ingestion design.

    uv run --group ingest scripts/analyze_techqa.py

The first pass gave us the schema. The open questions now are the ones that
change what we build:

  * Pooling. Each row ships only its own gold document, so retrieval within a
    row is trivial. The corpus only becomes a real retrieval task if every
    row's contexts are pooled into one index -- then finding the right doc
    among ~900 is genuine work. How many unique documents does that give?

  * The impossible rows. If an unanswerable question still carries a context,
    that context is related-but-not-answering. That is the hard refusal case
    and exactly what calibrates the threshold: retrieval WILL return something
    plausible and the system must still decline. If impossible rows have no
    contexts at all, they only exercise the easy path.

  * Document length. These look short (~1.3k chars). If most fit under the
    chunk ceiling, ingestion is one-chunk-per-document and the MDX chunker is
    not needed at all.
"""

from __future__ import annotations

import statistics
from collections import Counter

import pandas as pd

URL = "hf://datasets/nvidia/TechQA-RAG-Eval/train.json"


def main() -> int:
    df = pd.read_json(URL)
    print(f"rows: {len(df)}\n")

    imp = df["is_impossible"].astype(bool)
    print("=" * 60)
    print("ANSWERABLE VS IMPOSSIBLE")
    print("=" * 60)
    print(f"answerable  {(~imp).sum():>4}  ({(~imp).mean():.1%})")
    print(f"impossible  {imp.sum():>4}  ({imp.mean():.1%})")

    n_ctx = df["contexts"].map(len)
    print(
        f"\ncontexts per row: min {n_ctx.min()}, median "
        f"{int(n_ctx.median())}, max {n_ctx.max()}"
    )
    print(
        f"  answerable: median {int(n_ctx[~imp].median())}, "
        f"zero-context rows {(n_ctx[~imp] == 0).sum()}"
    )
    print(
        f"  impossible: median {int(n_ctx[imp].median())}, "
        f"zero-context rows {(n_ctx[imp] == 0).sum()}"
    )
    if imp.sum() and (n_ctx[imp] > 0).any():
        print(
            "\n  -> impossible rows DO carry documents: these are the hard"
            "\n     refusal cases (plausible retrieval, no answer present)."
        )
    else:
        print(
            "\n  -> impossible rows carry no documents: only the easy"
            "\n     refusal path is exercised."
        )

    print("\n" + "=" * 60)
    print("ANSWER FIELD")
    print("=" * 60)
    ans_len = df["answer"].map(lambda s: len(s.strip()))
    for label, mask in (("answerable", ~imp), ("impossible", imp)):
        sub = ans_len[mask]
        if not len(sub):
            continue
        print(
            f"{label:11} blank {int((sub == 0).sum()):>4}/{len(sub)}  "
            f"median {int(sub.median())} chars"
        )
    sample = df.loc[imp, "answer"].head(3).tolist()
    if sample:
        print("\nimpossible answer samples:")
        for s in sample:
            print(f"  {s[:100]!r}")

    print("\n" + "=" * 60)
    print("CORPUS AFTER POOLING")
    print("=" * 60)
    docs: dict[str, str] = {}
    repeats = Counter()
    for contexts in df["contexts"]:
        for c in contexts:
            name = c["filename"]
            repeats[name] += 1
            docs.setdefault(name, c["text"])

    print(f"context entries total : {sum(repeats.values())}")
    print(f"unique documents      : {len(docs)}")
    dupes = [n for n, c in repeats.items() if c > 1]
    print(f"documents reused      : {len(dupes)} (max reuse {max(repeats.values())})")

    lengths = sorted(len(t) for t in docs.values())
    words = sorted(len(t.split()) for t in docs.values())
    print(
        f"\ndoc chars  min {lengths[0]}  median "
        f"{int(statistics.median(lengths))}  p90 "
        f"{lengths[int(len(lengths) * 0.9)]}  max {lengths[-1]}"
    )
    print(
        f"doc words  min {words[0]}  median "
        f"{int(statistics.median(words))}  p90 "
        f"{words[int(len(words) * 0.9)]}  max {words[-1]}"
    )
    over = sum(1 for w in words if w > 800)
    print(
        f"docs over 800 words: {over}/{len(words)} "
        f"({over / len(words):.1%})  <- these need chunking"
    )

    print("\n" + "=" * 60)
    print("QUESTION SHAPE")
    print("=" * 60)
    qlen = df["question"].map(lambda s: len(s.split()))
    print(
        f"question words  median {int(qlen.median())}  "
        f"p90 {int(qlen.quantile(0.9))}  max {qlen.max()}"
    )
    print(
        "\nThese are forum posts, not search queries. Real support traffic"
        "\nlooks like this -- long and rambling -- which is a harder and"
        "\nmore honest retrieval test than the identifier eval."
    )
    print("\nsample question:")
    print("  " + df["question"].iloc[0][:300].replace("\n", " "))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
