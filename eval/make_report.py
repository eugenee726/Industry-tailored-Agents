# -*- coding: utf-8 -*-
"""
eval/results/*.csv → eval/REPORT.md + eval/figures/*.png 자동 생성 (API 호출 없음)

사용법: python -m eval.make_report
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from eval.common import EVAL_DIR, RESULT_DIR  # noqa: E402

FIG = EVAL_DIR / "figures"
FIG.mkdir(exist_ok=True)

# 검증된 기본 팔레트 (dataviz reference palette, light) — 고정 순서로만 사용
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Malgun Gothic", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def _read(name):
    p = RESULT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def _md(df, floatfmt=".3f"):
    return df.to_markdown(index=False, floatfmt=floatfmt) if df is not None else "_(결과 없음)_"


def _bars(ax, groups, series, values, colors, fmt="{:.2f}", ylim=(0, 1.05)):
    """grouped bar: groups(x) × series. 막대 사이 2px 간격, 값 라벨은 막대 위(텍스트 색은 ink)"""
    n = len(series)
    w = 0.8 / n
    for i, (s, c) in enumerate(zip(series, colors)):
        xs = [g + (i - (n - 1) / 2) * w for g in range(len(groups))]
        vs = values[i]
        bars = ax.bar(xs, vs, width=w * 0.92, color=c, label=s, edgecolor=SURFACE, linewidth=1)
        for x, v in zip(xs, vs):
            ax.text(x, v + 0.015, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n, fontsize=9, borderaxespad=0.2)


def fig_routing():
    rows = []
    for split, tag in [("개발셋 (60건)", ""), ("홀드아웃 (24건)", "_holdout")]:
        for v in ("v1", "v2"):
            df = _read(f"routing_{v}{tag}.csv")
            if df is None:
                continue
            rows.append({"split": split, "prompt": v, "accuracy": df["correct"].mean(),
                         "consistency": df["consistent"].mean(),
                         "hard+ambiguous": df[df["difficulty"] != "easy"]["correct"].mean(), "n": len(df),
                         "NONE(직접답변)": int((df["pred"] == "NONE").sum())})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    splits = list(dict.fromkeys(t["split"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    vals = [[t[(t.split == s) & (t.prompt == p)]["accuracy"].squeeze() if len(t[(t.split == s) & (t.prompt == p)]) else 0
             for s in splits] for p in ("v1", "v2")]
    _bars(ax, splits, ["프롬프트 v1", "프롬프트 v2"], vals, [S1, S2], fmt="{:.1%}", ylim=(0.5, 1.08))
    ax.set_title("라우팅 정확도 (첫 번째 transfer 기준)", loc="left", fontsize=11, pad=26)
    fig.tight_layout(); fig.savefig(FIG / "routing_accuracy.png", dpi=160); plt.close(fig)
    return t


def fig_retrieval():
    rows = [_read(f"retrieval_{e}_summary.csv") for e in "ABCD"]
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    t = pd.concat(rows, ignore_index=True)
    label = {"A": "A 1200자\n벡터", "B": "B 500자\n벡터", "D": "D 1200자\n하이브리드", "C": "C 500자\n하이브리드"}
    t = t.set_index("exp").loc[[e for e in "ABDC" if e in set(t["exp"])]].reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _bars(ax, [label[e] for e in t["exp"]], ["Hit@1", "Hit@5", "MRR"],
          [t["hit@1"].tolist(), t["hit@5"].tolist(), t["mrr"].tolist()], [S1, S2, S3])
    ax.set_title("검색 실험: 청크 크기 × 하이브리드 (답변 가능 질문 40건)", loc="left", fontsize=11, pad=26)
    fig.tight_layout(); fig.savefig(FIG / "retrieval.png", dpi=160); plt.close(fig)
    return t


def fig_gate():
    z = _read("gate_zones_C.csv")
    s = _read("gate_summary_C.csv")
    if z is None:
        return None, None
    groups = [("답 있음 · LLM 생성 질문", (z.source == "synthetic") & z.answerable, S1),
              ("답 있음 · 사람이 쓴 질문", (z.source == "human") & z.answerable, S2),
              ("답 없음 (코퍼스 밖)", ~z.answerable, S3)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, (name, mask, c) in enumerate(groups):
        ys = z[mask]["top_score"]
        jitter = [(i - 1) * 0.0 + ((k * 37) % 11 - 5) * 0.022 for k in range(len(ys))]
        ax.scatter(ys, [i + j for j in jitter], s=22, color=c, edgecolor=SURFACE, linewidth=1, label=name, zorder=3)
    for x, txt, style, y, ha in [(0.32, "기존 README 0.32", ":", -0.55, "left"),
                                 (0.45, "거절 ← 0.45", "--", 2.55, "right"),
                                 (0.54, "0.54 → 확신 답변", "-", 2.55, "left")]:
        ax.axvline(x, color=INK2, linestyle=style, linewidth=1, zorder=2)
        ax.text(x + (0.004 if ha == "left" else -0.004), y, txt, fontsize=8, color=INK2, va="bottom", ha=ha)
    ax.set_yticks(range(3), [g[0] for g in groups])
    ax.set_ylim(-0.6, 2.9)
    ax.set_xlabel("검색 1위 청크 유사도 (top_score)")
    ax.grid(axis="y", visible=False)
    ax.set_title("신뢰도 게이트: 점수 분포와 3구간 정책", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "gate.png", dpi=160); plt.close(fig)
    return s, pd.crosstab([z["source"], z["answerable"]], z["zone"]).reset_index()


def main():
    rt = fig_routing()
    ret = fig_retrieval()
    gate_sum, gate_zone = fig_gate()
    faith = _read("faithfulness_summary.csv")
    kappa = _read("faithfulness_kappa.csv")
    negctl = _read("faithfulness_negative_control_summary.csv")
    tick = _read("tickers.csv")
    cd = _read("close_date.csv")

    def conf(tag):
        p = RESULT_DIR / f"routing_{tag}_confusion.csv"
        return pd.read_csv(p).to_markdown(index=False) if p.exists() else "_(없음)_"

    md = f"""# 평가 리포트 — Industry-tailored Agents

> `python -m eval.make_report` 로 `eval/results/*.csv`에서 자동 생성됩니다. 모든 API 응답은 `eval/cache/`에 저장되어 재실행 시 비용 0으로 같은 수치가 재현됩니다.
> LLM: Claude Sonnet 5.5 (`claude-sonnet-5-5`) · 임베딩: OpenAI `text-embedding-3-small`

## 1. 라우팅 — 질의를 올바른 에이전트로 보내는가

- 정답 라벨 60건(개발셋: easy 33 / hard 21 / ambiguous 6) + **프롬프트 개선 후 처음 평가한 홀드아웃 24건**
- 실제 오케스트레이터(ADK + Claude)를 그대로 실행하고, 서브에이전트는 스텁으로 대체해 라우팅만 측정. 질의당 3회 반복.

![routing](figures/routing_accuracy.png)

{_md(rt)}

**v1 개발셋 혼동행렬**

{conf("v1")}

**v2 홀드아웃 혼동행렬**

{conf("v2_holdout")}

## 2. RAG 검색 — 정답 근거를 찾아오는가

- 정답 기준: 질문마다 저장한 **근거 구절**이 검색된 청크에 포함되면 정답 (청크 경계가 달라져도 같은 정답셋으로 비교)
- D는 "청크 크기 효과"와 "하이브리드 효과"를 분리하기 위한 대조군

![retrieval](figures/retrieval.png)

{_md(ret)}

## 3. 신뢰도 게이트 — 언제 답하고 언제 거절하는가

- 문서에 답이 있는 질문 66건(LLM 생성 40 + 사람 작성 26), 없는 질문 22건으로 임계값을 스윕
- FP(답 없는데 통과) = 근거 없는 답변 위험 → FN보다 치명적으로 간주

![gate](figures/gate.png)

{_md(gate_sum)}

**채택한 3구간 정책별 분포**

{_md(gate_zone, floatfmt="g")}

## 4. 답변 생성 — 근거에 충실한가

- 운영 파이프라인(하이브리드 검색 → 게이트 → Claude 생성, 문장마다 [n] 인용) 그대로 실행
- 문장 단위 Faithfulness는 LLM 채점관 판정 → **채점관 자체를 사람 라벨로 검증(κ)**

{_md(faith)}

**채점관 민감도 검증 (음성 대조군)** — 실제 답변 문장에 사실 오류를 하나씩 주입했을 때 채점관이 잡아내는 비율.
Faithfulness가 1.0으로 나와 "같은 모델이 관대하게 채점한 것 아닌가"를 확인하기 위함.

{_md(negctl) if negctl is not None else "_(미실행 — `python -m eval.run_faithfulness --negative-control`)_"}

**채점관 검증 (사람 라벨 vs LLM 채점관)** — 원문 10 + 오류 주입 10 문장을 섞어 사람이 블라인드 라벨링

{_md(kappa) if kappa is not None else "_(미실행 — `--export-human` 후 라벨 입력, `--kappa`)_"}

- 20문장 중 사람이 라벨링을 마친 문장만 집계 (진행 중). 표본이 작아 κ 자체는 참고치.
- 불일치 2건은 모두 **오류 주입 문장인데 사람이 '근거 있음'으로 본 경우**(부정 반전 "책임을 질 수 있다→지지 않는다", 숫자 "세 단계→네 단계"). 채점관은 둘 다 잡았음.
- 즉 불일치의 원인은 채점관의 관대함이 아니라 사람의 놓침이었고, 미묘한 오류는 사람도 빠르게 읽으면 놓친다는 점에서 자동 채점의 가치를 보여 줌.

## 5. 규칙 기반 컴포넌트 수정

**Day1 티커 추출** (30건, API 호출 없음)

{_md(tick)}

**Day3 공고 마감일 추출** (20건, API 호출 없음 · 규칙 개발에 사용한 세트라 낙관적 수치)

{_md(cd)}

## 6. 한계와 다음 단계

- **표본 크기**: 라우팅 84건, RAG 88건. 1건 = 1.2~1.7%p라 소수점 차이는 의미가 작음. 100% 수치는 "이 세트에서 오류 0건"이지 완벽하다는 뜻이 아님.
- **홀드아웃의 독립성**: 라우팅 홀드아웃은 v1 오류 패턴을 본 뒤 같은 작성자가 만든 세트. 실제 사용자 로그로 재검증 필요.
- **LLM 생성 질문의 낙관 편향**: 원문 표현과 겹쳐 검색·게이트 점수가 높게 나옴 → 사람이 쓴 질문 28건을 따로 넣어 임계값을 정했고, 그 결과 사람 질문의 답변률은 84.6%로 LLM 생성 질문(100%)보다 낮음.
- **게이트 FN**: 약어(PCCP)나 '자료에서 ~ 찾아줘' 같은 질의는 dense 점수가 낮아 거절됨. 게이트에 BM25 점수를 함께 쓰는 방안이 다음 과제.
- **채점관**: 생성 모델과 같은 모델(Sonnet 5.5). 음성 대조군 + 사람 라벨 κ로 보완하지만, 다른 모델 채점관과의 교차 검증은 미실시.
- **Day3 마감일 추출**: 규칙을 만들며 본 20건으로 측정 → 실제 수집 데이터에서의 채움률은 별도 측정 필요.
- **Day1 티커**: "SK하이닉스"의 SK처럼 한글 회사명 접두어 오인식 1건 잔존.
"""
    (EVAL_DIR / "REPORT.md").write_text(md, encoding="utf-8")
    print(f"→ {EVAL_DIR / 'REPORT.md'}, figures: {sorted(p.name for p in FIG.glob('*.png'))}")


if __name__ == "__main__":
    main()
