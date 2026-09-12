"""Small, deterministic BM25 retrieval over an explicitly supplied corpus."""

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold())


def load_training_corpus(
    path: str | Path = 'data/sft/train.jsonl',
) -> list[dict[str, str]]:
    """Read one JSONL file, retaining unique title/text pairs in first-seen order."""
    documents = []
    seen = set()
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for document in row['metadata']['documents']:
                key = (document['title'], document['text'])
                if key not in seen:
                    seen.add(key)
                    documents.append({'title': key[0], 'text': key[1]})
    return documents


class TinyBM25Retriever:
    """BM25 with positive IDF; unmatched documents are never returned.

    Repeated query tokens contribute repeatedly. Equal scores preserve input order.
    """

    def __init__(
        self,
        documents: Iterable[Mapping[str, str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.contents = tuple(d['title'] + '\n' + d['text'] for d in documents)
        self._frequencies = tuple(Counter(_tokenize(text)) for text in self.contents)
        self._lengths = tuple(sum(freq.values()) for freq in self._frequencies)
        count = len(self.contents)
        self._avgdl = sum(self._lengths) / count if count else 0.0
        document_frequency = Counter()
        for frequencies in self._frequencies:
            document_frequency.update(frequencies.keys())
        self._idf = {
            term: math.log1p((count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def retrieve(self, queries: list[str], topk: int = 3) -> dict:
        results = []
        for query in queries:
            if topk <= 0 or not self._avgdl:
                results.append([])
                continue
            terms = _tokenize(query)
            scored = []
            for index, frequencies in enumerate(self._frequencies):
                norm = self.k1 * (1 - self.b + self.b * self._lengths[index] / self._avgdl)
                score = 0.0
                for term in terms:
                    frequency = frequencies.get(term, 0)
                    if frequency:
                        score += self._idf[term] * frequency * (self.k1 + 1) / (frequency + norm)
                if score > 0:
                    scored.append((index, score))
            scored.sort(key=lambda item: -item[1])
            results.append([
                {'document': {'contents': self.contents[index]}}
                for index, _ in scored[:topk]
            ])
        return {'result': results}
