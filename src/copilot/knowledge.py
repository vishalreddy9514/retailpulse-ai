"""Retrieval (RAG) over the product catalogue and the project's documentation.

A small, dependency-free BM25 index. Lexical retrieval is deterministic, needs no
embedding service, and suits the data well: product descriptions are short and
keyword-like ("JUMBO BAG RED RETROSPOT"), and documentation questions use the same
vocabulary as the docs ("return policy", "lead time", "WAPE").

Documents:

* the product catalogue: one entry per stock code (canonical description);
* README sections, the inventory-assumptions file and the generated evaluation reports,
  split on Markdown headings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import config

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "s",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "which",
        "why",
        "with",
        "does",
        "do",
        "my",
        "our",
    }
)
DOC_FILES = [
    "README.md",
    "config/inventory_assumptions.toml",
    "reports/data_profile.md",
    "reports/forecast_evaluation.md",
    "reports/inventory_risk_evaluation.md",
]
MAX_CHUNK_CHARS = 1800


SUFFIXES = ("ing", "ed", "es", "s")


def stem(token: str) -> str:
    """Light suffix stripping so 'returns'/'return' and 'handled'/'handling' match."""
    for suffix in SUFFIXES:
        if len(token) > 4 and token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.tokens = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        df = Counter(t for toks in self.tokens for t in set(toks))
        n = len(self.tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(toks) for toks in self.tokens]

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, toks in zip(self.tf, self.tokens, strict=True):
            norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl)
            out.append(
                sum(
                    self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                    for t in q
                    if t in tf
                )
            )
        return out

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        ranked = sorted(range(len(s)), key=lambda i: (-s[i], i))
        return [(i, s[i]) for i in ranked[:k] if s[i] > 0]


@dataclass(frozen=True)
class DocChunk:
    source: str
    section: str
    text: str


def chunk_markdown(source: str, text: str) -> list[DocChunk]:
    """Split on headings; long sections are split further on blank lines."""
    chunks, heading, buf = [], source, []

    def flush():
        body = "\n".join(buf).strip()
        if not body:
            return
        while len(body) > MAX_CHUNK_CHARS:
            cut = body.rfind("\n\n", 0, MAX_CHUNK_CHARS)
            cut = cut if cut > 200 else MAX_CHUNK_CHARS
            chunks.append(DocChunk(source, heading, body[:cut].strip()))
            body = body[cut:].strip()
        chunks.append(DocChunk(source, heading, body))

    for line in text.splitlines():
        if re.match(r"^#{1,4} ", line):
            flush()
            heading, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    flush()
    return chunks


class KnowledgeBase:
    def __init__(self, products: pd.DataFrame, docs: list[DocChunk]):
        self.products = products.reset_index(drop=True)
        self.docs = docs
        self._product_index = BM25(
            (
                self.products["stock_code"] + " " + self.products["product_description"].fillna("")
            ).tolist()
        )
        # Section headings are repeated so a matching heading outweighs a passing mention.
        self._doc_index = BM25([f"{d.section}\n{d.section}\n{d.text}" for d in docs])

    @classmethod
    def build(
        cls, product_summary: pd.DataFrame, root: Path = config.PROJECT_ROOT
    ) -> KnowledgeBase:
        docs = []
        for rel in DOC_FILES:
            path = root / rel
            if path.exists():
                docs.extend(chunk_markdown(rel, path.read_text()))
        return cls(product_summary[["stock_code", "product_description"]], docs)

    def search_products(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        exact = self.products.index[self.products["stock_code"] == query.strip().upper()]
        if len(exact):
            return [(query.strip().upper(), float("inf"))][:k]
        return [
            (self.products.at[i, "stock_code"], s) for i, s in self._product_index.top(query, k)
        ]

    def search_docs(self, query: str, k: int = 4) -> list[tuple[DocChunk, float]]:
        return [(self.docs[i], s) for i, s in self._doc_index.top(query, k)]
