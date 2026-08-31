#!/usr/bin/env python
"""Inspect TechQA-RAG-Eval before committing to an ingestion path.

    uv run --group ingest scripts/inspect_techqa.py

Deliberately schema-agnostic: it prints what is actually there rather than
assuming field names. The three things that decide the ingestion design are

  1. Where the document text lives -- inline per question, or a separate
     corpus file. Inline means duplicated documents to dedupe on load.
  2. How unanswerable questions are marked. These are what calibrate the
     refusal threshold, so if they are not clearly flagged the dataset is
     much less useful than advertised.
  3. Corpus size. A few hundred docs slots into the existing chunks table
     under a different `corpus` value; tens of thousands needs its own path
     and probably an ANN index.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import pandas as pd

URL = "hf://datasets/nvidia/TechQA-RAG-Eval/train.json"
MAX_PREVIEW = 400


def describe(value, depth: int = 0) -> str:
    pad = "  " * depth
    if isinstance(value, dict):
        lines = [f"{pad}dict({len(value)} keys)"]
        for k, v in list(value.items())[:8]:
            lines.append(f"{pad}  {k}: {describe(v, depth + 2).strip()}")
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        inner = describe(value[0], depth + 1).strip() if value else "empty"
        return f"{pad}list[{len(value)}] of {inner}"
    if isinstance(value, str):
        preview = value[:120].replace("\n", " ")
        return f"{pad}str({len(value)} chars) {preview!r}"
    return f"{pad}{type(value).__name__}({value!r})"


def main() -> int:
    print(f"loading {URL}\n")
    try:
        df = pd.read_json(URL)
    except Exception as exc:  # noqa: BLE001
        print(f"failed: {exc}\n")
        print("If this is an auth error: pip install huggingface_hub, then")
        print("  hf auth login        (older versions: huggingface-cli login)")
        return 1

    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}\n")

    print("=" * 60)
    print("COLUMN TYPES AND FILL")
    print("=" * 60)
    for col in df.columns:
        non_null = df[col].notna().sum()
        kinds = Counter(type(v).__name__ for v in df[col].head(200))
        print(f"{col:28} {non_null:>5}/{len(df)} non-null  {dict(kinds)}")

    print("\n" + "=" * 60)
    print("FIRST RECORD, FIELD BY FIELD")
    print("=" * 60)
    row = df.iloc[0]
    for col in df.columns:
        print(f"\n--- {col}")
        print(describe(row[col]))

    print("\n" + "=" * 60)
    print("LOOKING FOR UNANSWERABLE FLAGS")
    print("=" * 60)
    # Any low-cardinality or boolean column is a candidate label.
    for col in df.columns:
        try:
            uniques = df[col].dropna().unique()
        except TypeError:
            continue  # unhashable (lists/dicts) -- not a label column
        if 1 < len(uniques) <= 12:
            print(f"{col}: {list(uniques)[:12]}")
        elif len(uniques) == 1:
            print(f"{col}: constant = {uniques[0]!r}")

    # Empty or sentinel answers are the other common way to mark these.
    for col in df.columns:
        sample = df[col].head(200)
        if sample.map(lambda v: isinstance(v, str)).all():
            blank = sample.map(lambda v: not v.strip()).sum()
            if blank:
                print(f"{col}: {blank}/200 blank strings "
                      f"(possible unanswerable marker)")

    print("\n" + "=" * 60)
    print("DOCUMENT TEXT LOCATION")
    print("=" * 60)
    for col in df.columns:
        first = df[col].iloc[0]
        if isinstance(first, str) and len(first) > 500:
            lengths = df[col].map(lambda v: len(v) if isinstance(v, str) else 0)
            print(f"{col}: long text, median {int(lengths.median())} chars "
                  f"-- likely document body")
        if isinstance(first, list) and first and isinstance(first[0], (dict, str)):
            counts = df[col].map(lambda v: len(v) if isinstance(v, list) else 0)
            print(f"{col}: list per row, median {int(counts.median())} items "
                  f"-- likely candidate documents")

    out = "data/techqa_sample.json"
    df.head(3).to_json(out, orient="records", indent=2)
    print(f"\nfirst 3 rows written to {out} -- open it and read one end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())