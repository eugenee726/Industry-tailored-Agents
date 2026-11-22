# -*- coding: utf-8 -*-
"""
문자 bigram BM25 + RRF 결합 (외부 의존성 없음)

- 한국어 PDF 추출 텍스트는 띄어쓰기가 깨져 있어(예: '식약처는2025년5월7일자로') 단어 단위 BM25가 잘 안 맞음
  → 공백을 제거한 문자 bigram을 토큰으로 사용
- eval/run_retrieval.py 실험 C(500자 청크 + 하이브리드)가 MRR 0.632 → 0.844로 가장 좋아 운영 기본값으로 채택
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

RRF_K = 60


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "").lower()


class CharBM25:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(self._grams(t)) for t in texts]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df = Counter(g for d in self.docs for g in d)
        n = len(self.docs)
        self.idf = {g: math.log(1 + (n - c + 0.5) / (c + 0.5)) for g, c in df.items()}
        self.k1, self.b = k1, b

    @staticmethod
    def _grams(t: str) -> List[str]:
        t = _norm(t)
        return [t[i:i + 2] for i in range(len(t) - 1)]

    def scores(self, query: str) -> np.ndarray:
        qg = Counter(self._grams(query))
        out = np.zeros(len(self.docs))
        for j, (d, L) in enumerate(zip(self.docs, self.lens)):
            s = 0.0
            for g in qg:
                f = d.get(g, 0)
                if f:
                    s += self.idf[g] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out[j] = s
        return out


def rrf_merge(*rankings: Sequence[str], k: int = RRF_K) -> List[str]:
    """여러 순위 리스트(문서 id)를 Reciprocal Rank Fusion으로 결합"""
    score: Dict[str, float] = Counter()
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            score[doc_id] += 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(score.items(), key=lambda x: -x[1])]
