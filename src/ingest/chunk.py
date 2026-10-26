from __future__ import annotations

import re
from dataclasses import dataclass, field

from .mdx import Doc, Section

MAX_WORDS = 800
MIN_WORDS = 150
OVERLAP_SENTENCES = 2
ROOT_CRUMB = "Chatwoot Docs"

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CODE_TOKEN_RE = re.compile(r"\x00CODE\d+\x00")


@dataclass
class Chunk:
    doc_path: str
    breadcrumb: list[str]
    heading_level: int
    content: str
    identifiers: list[str] = field(default_factory=list)

    @property
    def crumb_text(self) -> str:
        return " > ".join(self.breadcrumb)

    def word_count(self) -> int:
        return words(self.content)


def words(text: str) -> int:
    return len(text.split())


def section_words(section: Section) -> int:
    total = words(section.body_text()) + words(section.heading)
    return total + sum(section_words(c) for c in section.children)


def flatten(section: Section, include_heading: bool = False) -> str:
    parts: list[str] = []
    if include_heading and section.heading:
        parts.append(section.heading)
    body = section.body_text()
    if body:
        parts.append(body)
    for child in section.children:
        parts.append(flatten(child, include_heading=True))
    return "\n\n".join(p for p in parts if p.strip())


def tail_sentences(text: str, n: int) -> str:
    """Last n sentences, for overlap. Never carries a code block across."""
    sentences = SENTENCE_RE.split(text.strip())
    tail = " ".join(sentences[-n:]).strip()
    if CODE_TOKEN_RE.search(tail):
        return ""
    return tail


def hard_split(text: str, max_words: int = MAX_WORDS) -> list[str]:
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    count = 0

    for para in paragraphs:
        w = words(para)
        if current and count + w > max_words:
            joined = "\n\n".join(current)
            parts.append(joined)
            overlap = tail_sentences(joined, OVERLAP_SENTENCES)
            current = [overlap] if overlap else []
            count = words(overlap)
        current.append(para)
        count += w

    if current:
        parts.append("\n\n".join(current))
    return parts


def emit(
    section: Section,
    trail: list[str],
    doc_path: str,
    chunks: list[Chunk],
    boundary: int = 2,
) -> None:
    crumb = trail + [section.heading] if section.heading else list(trail)

    if section.level >= boundary and section_words(section) <= MAX_WORDS:
        text = flatten(section)
        if text.strip():
            chunks.append(Chunk(doc_path, crumb, section.level, text))
        return

    body = section.body_text()
    if body:
        pieces = hard_split(body) if words(body) > MAX_WORDS else [body]
        for piece in pieces:
            chunks.append(Chunk(doc_path, crumb, section.level, piece))

    for child in section.children:
        emit(child, crumb, doc_path, chunks, boundary)


def common_prefix(a: list[str], b: list[str]) -> list[str]:
    out: list[str] = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out or a[:1]


def combine(a: Chunk, b: Chunk) -> Chunk:
    crumb = common_prefix(a.breadcrumb, b.breadcrumb)
    leaf = b.breadcrumb[len(crumb) :]
    header = " > ".join(leaf)
    body = (
        f"{a.content}\n\n{header}\n{b.content}"
        if header
        else f"{a.content}\n\n{b.content}"
    )
    return Chunk(a.doc_path, crumb, min(a.heading_level, b.heading_level), body)


def merge_small(chunks: list[Chunk]) -> list[Chunk]:
    """Merge undersized chunks forward. Called per document only."""
    out: list[Chunk] = []
    pending: Chunk | None = None

    for chunk in chunks:
        if pending is not None:
            chunk = combine(pending, chunk)
            pending = None
        if chunk.word_count() < MIN_WORDS:
            pending = chunk
            continue
        out.append(chunk)

    if pending is not None:
        if out:
            out[-1] = combine(out[-1], pending)
        else:
            out.append(pending)
    return out


def chunk_doc(doc: Doc) -> list[Chunk]:
    chunks: list[Chunk] = []
    emit(doc.root, [ROOT_CRUMB], doc.path, chunks)
    return merge_small(chunks)
