# -*- coding: utf-8 -*-
"""
Day3 공고 마감일 추출 평가 (API 호출 없음)
- before: 기존 normalize는 close_date를 항상 "" → 채움률 0%
- after : extract_close_date 규칙 기반 추출
지표: 채움률(정답에 마감일이 있는 공고 중 추출 성공 비율), 정확도(추출값이 정답과 일치), 오탐(없는데 추출)
"""
from __future__ import annotations

from eval.common import DATA_DIR, load_jsonl, save_csv


def evaluate(rows, fn):
    has = [r for r in rows if r["gold"]]
    none = [r for r in rows if not r["gold"]]
    pred = {id(r): fn(r) for r in rows}
    filled = sum(bool(pred[id(r)]) for r in has)
    correct = sum(pred[id(r)] == r["gold"] for r in has)
    fp = sum(bool(pred[id(r)]) for r in none)
    return {"fill_rate": round(filled / len(has), 3), "accuracy_when_present": round(correct / len(has), 3),
            "false_extractions": fp, "n_with_deadline": len(has), "n_without": len(none)}


def main():
    from student.day3.impl.normalize import extract_close_date
    rows = load_jsonl(DATA_DIR / "close_dates.jsonl")
    res = [{"version": "before", **evaluate(rows, lambda r: "")},
           {"version": "after", **evaluate(rows, lambda r: extract_close_date(r["text"], r["ref_year"]))}]
    save_csv(res, "close_date.csv")
    for r in rows:
        got = extract_close_date(r["text"], r["ref_year"])
        if got != r["gold"]:
            print(f"  MISMATCH gold={r['gold']!r} got={got!r} :: {r['text']}")
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
