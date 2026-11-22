# -*- coding: utf-8 -*-
"""
RAG 신뢰도 게이트 임계값 분석 (API 호출 없음; run_retrieval 결과 사용)

데이터 (라벨 answerable: 1=문서에 답 있음, 0=없음)
  - synthetic : LLM이 청크에서 만든 질문 40 (전부 1) + 사람이 쓴 코퍼스 밖 질문 20 (전부 0)
  - human     : 사람이 쓴 실제 문체의 문서 질문 28 (26=1, 2=0)
  ※ synthetic만으로 정하면 임계값이 과도하게 높아짐(원문 표현과 겹쳐 점수가 높음) → 반드시 합쳐서 결정

오류의 비용
  - FP(답 없는데 통과): 근거 없는 답변 생성 위험 → 치명적
  - FN(답 있는데 차단): "근거 부족" 응답 → 불편하지만 안전

정책: 단일 임계값으로는 두 분포가 겹쳐 완벽 분리가 불가 → 3구간
  - top_score ≥ T_HIGH : 확신 답변 (이 구간에 '답 없는 질문'이 하나도 없도록 하는 가장 낮은 값, Precision=1.0)
  - T_LOW ≤ top < T_HIGH: '근거 약함' 경고 + 답변 + 웹 보강 권고
  - top_score < T_LOW  : 거절 (답 있는 질문의 95% 이상이 T_LOW 위에 오도록 하는 가장 높은 값)

사용법: python -m eval.run_gate --exp C
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from eval.common import RESULT_DIR, save_csv

MIN_PRECISION = 0.95       # 단일 임계값 비교용
MIN_RECALL_FOR_LOW = 0.95


def evaluate(df, t_top, t_mean=0.0):
    pred = (df["top_score"] >= t_top) & (df["mean_top5"] >= t_mean)
    y = df["answerable"].astype(bool)
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"t_top": round(float(t_top), 2), "t_mean": round(float(t_mean), 2), "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="C")
    args = ap.parse_args()
    syn = pd.read_csv(RESULT_DIR / f"retrieval_{args.exp}.csv").assign(source="synthetic")
    hum = pd.read_csv(RESULT_DIR / f"retrieval_{args.exp}_human.csv").assign(source="human")
    df = pd.concat([syn, hum], ignore_index=True)
    grid = np.round(np.arange(0.10, 0.71, 0.01), 2)

    print("top_score 분포")
    print(df.groupby(["source", "answerable"])["top_score"].describe()[["count", "mean", "min", "50%", "max"]].round(3).to_string())

    # 1) 단일 임계값 곡선 (synthetic만 vs 합친 데이터)
    curve_all = [evaluate(df, t) | {"data": "all"} for t in grid]
    curve_syn = [evaluate(syn, t) | {"data": "synthetic_only"} for t in grid]
    save_csv(curve_all + curve_syn, f"gate_curve_{args.exp}.csv")

    # 2) 기존 설정 평가 (AND 규칙)
    refs = []
    for name, (a, b) in {"code(0.20/0.20)": (0.20, 0.20), "readme(0.32/0.30)": (0.32, 0.30)}.items():
        refs.append({"setting": name, **evaluate(df, a, b)})

    # 3) synthetic만으로 정했을 때의 함정
    syn_ok = [c for c in curve_syn if c["precision"] >= MIN_PRECISION]
    syn_pick = max(syn_ok, key=lambda c: (c["recall"], -c["t_top"]))
    syn_pick_on_all = evaluate(df, syn_pick["t_top"])

    # 4) 3구간 정책
    single = min(c["t_top"] for c in curve_all if c["precision"] >= MIN_PRECISION and c["recall"] >= MIN_RECALL_FOR_LOW)
    high = min(c["t_top"] for c in curve_all if c["fp"] == 0)
    low = max(c["t_top"] for c in curve_all if c["recall"] >= MIN_RECALL_FOR_LOW)
    zone = np.where(df["top_score"] >= high, "answer", np.where(df["top_score"] >= low, "warn", "refuse"))
    df["zone"] = zone
    zt = pd.crosstab([df["source"], df["answerable"]], df["zone"]).reindex(columns=["answer", "warn", "refuse"], fill_value=0)
    save_csv(df.to_dict("records"), f"gate_zones_{args.exp}.csv")

    summary = refs + [
        {"setting": f"synthetic로만 선정({syn_pick['t_top']}) → 전체 데이터", **syn_pick_on_all},
        {"setting": f"단일 임계값 선정({single})", **evaluate(df, single)},
        {"setting": f"3구간: 확신 답변 구간만(T_HIGH={high})", **evaluate(df, high)},
        {"setting": f"3구간: 거절 제외 전체(T_LOW={low})", **evaluate(df, low)},
    ]
    save_csv(summary, f"gate_summary_{args.exp}.csv")
    print("\n설정별 성능 (전체 88건 기준, 통과=답변)")
    print(pd.DataFrame(summary)[["setting", "precision", "recall", "fp", "fn"]].to_string(index=False))
    print(f"\n3구간 정책: T_HIGH={high} (FP 0), T_LOW={low} (답 있는 질문 {int(MIN_RECALL_FOR_LOW*100)}% 이상 통과)")
    print(zt.to_string())
    print("\n'refuse'로 잘못 막힌 답 있는 질문:")
    for _, r in df[(df["zone"] == "refuse") & df["answerable"]].iterrows():
        print(f"  {r['id']} top={r['top_score']:.3f} {r['query']}")
    print("'warn' 구간의 답 없는 질문 (경고와 함께 답변됨):")
    for _, r in df[(df["zone"] != "refuse") & ~df["answerable"]].iterrows():
        print(f"  {r['id']} top={r['top_score']:.3f} {r['query']}")


if __name__ == "__main__":
    main()
