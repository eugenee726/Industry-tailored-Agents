# -*- coding: utf-8 -*-
"""평가 지표 (numpy/pandas만 사용)"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def classification_report(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> pd.DataFrame:
    rows = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"label": lab, "precision": prec, "recall": rec, "f1": f1, "support": tp + fn})
    return pd.DataFrame(rows)


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0


def confusion(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> pd.DataFrame:
    pred_labels = list(labels) + sorted({p for p in y_pred if p not in labels})
    m = pd.DataFrame(0, index=list(labels), columns=pred_labels)
    for t, p in zip(y_true, y_pred):
        m.loc[t, p] += 1
    m.index.name = "true \\ pred"
    return m


def hit_at_k(ranks: List[int | None], k: int) -> float:
    """ranks: 정답이 처음 등장한 순위(1-based), 없으면 None"""
    return float(np.mean([r is not None and r <= k for r in ranks])) if ranks else 0.0


def mrr(ranks: List[int | None]) -> float:
    return float(np.mean([1.0 / r if r else 0.0 for r in ranks])) if ranks else 0.0


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return 0.0
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def pr_at_threshold(scores: Sequence[float], labels: Sequence[int], thr: float) -> Dict[str, float]:
    """labels: 1 = 답변 가능(문서에 근거 있음), 0 = 답변 불가. score >= thr 이면 '통과' 예측"""
    pred = [s >= thr for s in scores]
    tp = sum(p and l for p, l in zip(pred, labels))
    fp = sum(p and not l for p, l in zip(pred, labels))
    fn = sum((not p) and l for p, l in zip(pred, labels))
    tn = sum((not p) and (not l) for p, l in zip(pred, labels))
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": thr, "precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
