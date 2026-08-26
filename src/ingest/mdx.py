"""Parse Mintlify MDX into a heading tree.

Design notes
------------
Fenced code blocks are pulled out FIRST and replaced with placeholders. Two
reasons:

1. Deployment docs are full of bash comments like `# Install Chatwoot`. Without
   this, those get parsed as H1 headings and the section tree is nonsense.
2. Chunking then operates on prose-only word counts, and code can never be
   split across a chunk boundary — it is a single opaque token until the very
   last step.

Placeholders are restored after chunking, in run_ingest.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

FENCE_RE = re.compile(r"^\s*(```|~~~)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
IMPORT_RE = re.compile(r"^\s*(?:import|export)\s+.*$", re.MULTILINE)
MDX_COMMENT_RE = re.compile(r"\{/\*.*?\*/\}", re.DOTALL)
JSX_TAG_RE = re.compile(r"</?[A-Z][A-Za-z0-9.]*(?:\s[^>]*?)?/?>", re.DOTALL)
JSX_ATTR_RE = re.compile(r"""\b(?:title|label|header|name)\s*=\s*["']([^"']*)["']""")
CODE_PLACEHOLDER = "\x00CODE{}\x00"
CODE_PLACEHOLDER_RE = re.compile(r"\x00CODE(\d+)\x00")


@dataclass
class Section:
    level: int
    heading: str
    body: list[str] = field(default_factory=list)
    children: list[Section] = field(default_factory=list)

    def body_text(self) -> str:
        return "\n".join(self.body).strip()


@dataclass
class Doc:
    path: str
    title: str
    description: str
    root: Section
    code_blocks: list[str]


def extract_code(text: str) -> tuple[str, list[str]]:
    """Replace fenced blocks with placeholders. Returns (text, blocks)."""
    out: list[str] = []
    blocks: list[str] = []
    buf: list[str] = []
    fence: str | None = None

    for line in text.split("\n"):
        m = FENCE_RE.match(line)
        if fence is None:
            if m:
                fence = m.group(1)
                buf = [line]
            else:
                out.append(line)
        else:
            buf.append(line)
            if m and m.group(1) == fence:
                out.append(CODE_PLACEHOLDER.format(len(blocks)))
                blocks.append("\n".join(buf))
                fence, buf = None, []

    if fence is not None:  # unterminated fence — keep it rather than lose it
        out.append(CODE_PLACEHOLDER.format(len(blocks)))
        blocks.append("\n".join(buf))

    return "\n".join(out), blocks


def restore_code(text: str, blocks: list[str]) -> str:
    return CODE_PLACEHOLDER_RE.sub(lambda m: blocks[int(m.group(1))], text)


def strip_jsx(text: str) -> str:
    """Remove JSX tags, keeping inner content and any title/label attribute.

    `<Step title="Subscribe to Chatwoot">` carries real information, so the
    attribute is promoted to a text line rather than discarded with the tag.
    """
    text = MDX_COMMENT_RE.sub("", text)
    text = IMPORT_RE.sub("", text)

    def _replace(m: re.Match) -> str:
        tag = m.group(0)
        if tag.startswith("</"):
            return ""
        attrs = JSX_ATTR_RE.findall(tag)
        return ("\n" + " ".join(attrs) + "\n") if attrs else ""

    text = JSX_TAG_RE.sub(_replace, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def same_heading(a: str, b: str) -> bool:
    """Compare headings ignoring case, punctuation and backticks."""
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return bool(a) and bool(b) and norm(a) == norm(b)


def build_tree(text: str, doc_title: str) -> Section:
    root = Section(level=0, heading=doc_title)
    stack = [root]

    for line in text.split("\n"):
        m = HEADING_RE.match(line)
        if not m:
            stack[-1].body.append(line)
            continue

        level = len(m.group(1))
        heading = strip_jsx(m.group(2)).strip()
        heading = re.sub(r"[`*_]", "", heading).strip()

        while len(stack) > 1 and stack[-1].level >= level:
            stack.pop()

        # Many files repeat the frontmatter title as an H1, which would give
        # breadcrumbs like "Common Errors > Common Errors". Fold a heading into
        # its parent instead of nesting a duplicate node. Pop first, so the
        # comparison is against the real parent at this level.
        if same_heading(stack[-1].heading, heading):
            continue

        section = Section(level=level, heading=heading)
        stack[-1].children.append(section)
        stack.append(section)

    return root


def parse_file(path: Path, corpus_root: Path) -> Doc:
    post = frontmatter.load(path)
    rel = str(path.relative_to(corpus_root))

    title = str(post.get("title") or path.stem.replace("-", " ").title())
    description = str(post.get("description") or "")

    text, blocks = extract_code(post.content)
    text = strip_jsx(text)
    root = build_tree(text, title)

    if description:
        root.body.insert(0, description)

    return Doc(
        path=rel, title=title, description=description, root=root, code_blocks=blocks
    )
