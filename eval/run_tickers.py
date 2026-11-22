# -*- coding: utf-8 -*-
"""
Day1 티커 추출 평가 (API 호출 없음)
- before: 수정 전 구현(query.upper() 후 대문자 1~5자 전부 추출)을 그대로 보존해 비교
- after : student/day1/agent.py 의 현재 구현
지표: 티커 단위 Precision / Recall, 질의 단위 완전일치율, 오탐(false positive) 질의 수
"""
from __future__ import annotations

import re
from typing import List

from eval.common import DATA_DIR, load_jsonl, save_csv


def extract_before(query: str) -> List[str]:
    cleaned = re.sub(r"[\/,\|]", " ", query.upper())
    hits = re.findall(r"\b[A-Z]{1,5}\b", cleaned) + re.findall(r"\b\d{6}\b", cleaned)
    out: List[str] = []
    for h in hits:
        if h not in out:
            out.append(h)
    return out


def score(rows, fn):
    tp = fp = fn_ = exact = fp_queries = 0
    for r in rows:
        pred, gold = set(fn(r["query"])), set(r["gold"])
        tp += len(pred & gold); fp += len(pred - gold); fn_ += len(gold - pred)
        exact += pred == gold
        fp_queries += bool(pred - gold)
    p = tp / (tp + fp) if tp + fp else 1.0
    rc = tp / (tp + fn_) if tp + fn_ else 1.0
    return {"precision": round(p, 3), "recall": round(rc, 3), "exact_match": round(exact / len(rows), 3),
            "false_positive_queries": fp_queries, "n": len(rows)}


def main():
    from student.day1.agent import _extract_tickers_from_query as extract_after
    rows = load_jsonl(DATA_DIR / "tickers.jsonl")
    res = [{"version": "before", **score(rows, extract_before)},
           {"version": "after", **score(rows, extract_after)}]
    save_csv(res, "tickers.csv")
    detail = [{"query": r["query"], "gold": r["gold"], "before": extract_before(r["query"]),
               "after": extract_after(r["query"])} for r in rows]
    save_csv(detail, "tickers_detail.csv")
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
