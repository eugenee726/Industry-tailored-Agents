# -*- coding: utf-8 -*-
"""
RAG 답변 품질 평가: 근거 충실도(Faithfulness), 인용 유효성, 거절 정확도, 정답성

파이프라인은 운영과 동일: Day2Agent.retrieve(하이브리드) → _gate(3구간) → generate_answer(Claude, [n] 인용)
데이터: rag_qa(LLM 생성 답변가능 40 + 코퍼스 밖 20) + gate_human(사람이 쓴 문서 질문 28)

지표
  - 답변률(답 있는 질문)       : 게이트 통과 & LLM이 거절하지 않은 비율
  - 거절 정확도(답 없는 질문)   : 게이트 거절 또는 LLM이 "근거를 찾지 못함"으로 거절한 비율
  - 인용 유효율               : 답변 속 [n]이 실제 제공된 컨텍스트 번호(1..K) 안에 있는 비율 (규칙 기반)
  - 인용 커버리지             : 인용이 붙은 문장 비율 (규칙 기반)
  - Faithfulness              : 문장 단위로 'context가 뒷받침하는가'를 LLM 채점관이 판정한 비율
  - 정답성(synthetic만)        : 근거 구절(evidence)에 비추어 질문에 맞게 답했는지 (LLM 채점관)

채점관 검증: --export-human 으로 문장 20개를 뽑아 사람이 라벨 → --kappa 로 채점관-사람 일치도(Cohen's κ)

사용법:
  python -m eval.run_faithfulness                 # 생성 + 채점
  python -m eval.run_faithfulness --export-human  # 사람 라벨용 CSV 생성
  python -m eval.run_faithfulness --kappa         # 사람 라벨과 채점관 일치도
"""
from __future__ import annotations

import argparse
import random
import re

import pandas as pd

from eval.common import DATA_DIR, RESULT_DIR, JsonlCache, load_jsonl, save_csv
from eval.metrics import cohen_kappa

GEN_VERSION = "v1"
HUMAN_LABEL_FILE = DATA_DIR / "faithfulness_human_labels.csv"

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "supported", "reason"],
                "additionalProperties": False,
            },
        },
        "answers_question": {"type": "boolean"},
    },
    "required": ["sentences", "answers_question"],
    "additionalProperties": False,
}

JUDGE_PROMPT = """당신은 RAG 답변을 검수하는 엄격한 채점관입니다.

<context>
{context}
</context>

<question>{question}</question>
{evidence_block}
<answer_sentences>
{sentences}
</answer_sentences>

1) 각 문장(index)이 context의 내용으로 **직접** 뒷받침되면 supported=true, context에 없는 내용·과장·추측이 섞이면 false.
   (인용 번호 [n] 표기 자체는 판단에서 무시하고 내용만 보세요. reason은 한 문장.)
2) answers_question: 답변이 질문에 실질적으로 답했으면 true.{evidence_rule}"""


def split_sentences(answer: str):
    # "...입니다. [1][4]"처럼 마침표 뒤에 붙은 인용을 앞 문장 안으로 옮긴 뒤 분리
    text = re.sub(r"([.!?。])\s*((?:\[\d+\])+)", r" \2\1", answer.strip())
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text)
    return [p.strip(" -•*") for p in parts if len(p.strip(" -•*")) > 3]


def citation_stats(answer: str, k: int):
    cites = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    valid = sum(1 <= c <= k for c in cites)
    sents = split_sentences(answer)
    covered = sum(bool(re.search(r"\[\d+\]", s)) for s in sents)
    return {"n_cites": len(cites), "valid_cites": valid,
            "n_sentences": len(sents), "cited_sentences": covered}


def load_data():
    rows = load_jsonl(DATA_DIR / ("rag_qa.jsonl" if (DATA_DIR / "rag_qa.jsonl").exists() else "rag_qa.draft.jsonl"))
    for r in rows:
        r["source"] = "synthetic" if r["answerable"] else "out_of_scope"
    for r in load_jsonl(DATA_DIR / "gate_human.jsonl"):
        r["source"] = "human"
        rows.append(r)
    return rows


def run_pipeline():
    from student.common.schemas import Day2Plan
    from student.common.llm import claude_json
    from student.day2.impl.embeddings import Embeddings
    from student.day2.impl.rag import NO_EVIDENCE, Day2Agent, _gate, generate_answer

    plan = Day2Plan()
    agent = Day2Agent(plan)
    emb = Embeddings(model=plan.embedding_model)
    gen_cache, judge_cache = JsonlCache("rag_generation"), JsonlCache("rag_judge")
    data = load_data()
    print(f"[faithfulness] 질문 {len(data)}건 (게이트 통과분만 생성·채점)")

    rows, sent_rows = [], []
    for d in data:
        contexts, dense_scores = agent.retrieve(d["query"], plan, emb)
        gate = _gate(dense_scores, plan)
        ctx_ids = [c["doc_id"] for c in contexts]
        answer, refused_by = "", ""
        if gate["zone"] == "refuse":
            refused_by = "gate"
        else:
            answer = gen_cache.get_or_call({"v": GEN_VERSION, "q": d["query"], "ctx": ctx_ids},
                                           lambda: generate_answer(d["query"], contexts))
            if NO_EVIDENCE in answer and len(answer) < len(NO_EVIDENCE) + 15:
                refused_by = "llm"

        row = {"id": d["id"], "source": d["source"], "query": d["query"], "answerable": d["answerable"],
               "zone": gate["zone"], "top_score": round(gate["top_score"], 3), "refused_by": refused_by,
               "answer": answer}
        if answer and not refused_by:
            row.update(citation_stats(answer, len(contexts)))
            sents = split_sentences(answer)
            ctx_text = "\n\n".join(f"[{i}] {re.sub(r'\s+', ' ', c['chunk'])}" for i, c in enumerate(contexts, 1))
            ev = d.get("evidence") or ""
            prompt = JUDGE_PROMPT.format(
                context=ctx_text, question=d["query"],
                evidence_block=f"<reference_evidence>{ev}</reference_evidence>\n" if ev else "",
                sentences="\n".join(f"{i}. {s}" for i, s in enumerate(sents)),
                evidence_rule=" reference_evidence가 주어졌다면 그 내용과 일치하는 답을 했을 때만 true." if ev else "")
            j = judge_cache.get_or_call({"v": 2, "q": d["query"], "answer": answer, "ctx": ctx_ids},
                                        lambda: claude_json(prompt, JUDGE_SCHEMA, max_tokens=3000))
            sup = {s["index"]: s for s in j["sentences"]}
            for i, s in enumerate(sents):
                js = sup.get(i, {"supported": False, "reason": "judge 누락"})
                sent_rows.append({"qid": d["id"], "source": d["source"], "sent_idx": i, "sentence": s,
                                  "judge_supported": js["supported"], "judge_reason": js["reason"]})
            row["n_supported"] = sum(sup.get(i, {}).get("supported", False) for i in range(len(sents)))
            row["answers_question"] = j["answers_question"]
        rows.append(row)
    print(gen_cache.stats(), judge_cache.stats())

    df = save_csv(rows, "faithfulness_answers.csv")
    sdf = save_csv(sent_rows, "faithfulness_sentences.csv")
    summarize(df, sdf)


def summarize(df, sdf):
    out = []
    for src, g in [("all", df)] + list(df.groupby("source")):
        ans, una = g[g["answerable"]], g[~g["answerable"]]
        answered = g[(g["answer"].fillna("") != "") & (g["refused_by"].fillna("") == "")]
        s = sdf[sdf["qid"].isin(answered["id"])]
        rec = {
            "subset": src, "n": len(g),
            "answer_rate(answerable)": round((ans["refused_by"].fillna("") == "").mean(), 3) if len(ans) else None,
            "refusal_acc(unanswerable)": round((una["refused_by"].fillna("") != "").mean(), 3) if len(una) else None,
            "refused_by_gate": int((g["refused_by"] == "gate").sum()),
            "refused_by_llm": int((g["refused_by"] == "llm").sum()),
            "faithfulness(sentence)": round(s["judge_supported"].mean(), 3) if len(s) else None,
            "fully_faithful_answers": round((answered["n_supported"] == answered["n_sentences"]).mean(), 3) if len(answered) else None,
            "citation_validity": round(answered["valid_cites"].sum() / max(1, answered["n_cites"].sum()), 3) if len(answered) else None,
            "citation_coverage": round(answered["cited_sentences"].sum() / max(1, answered["n_sentences"].sum()), 3) if len(answered) else None,
            "answers_question": round(answered["answers_question"].mean(), 3) if len(answered) else None,
        }
        out.append(rec)
    res = pd.DataFrame(out)
    save_csv(out, "faithfulness_summary.csv")
    print(res.T.to_string())
    bad = sdf[~sdf["judge_supported"]]
    if len(bad):
        print(f"\n근거 없음으로 판정된 문장 {len(bad)}개:")
        for _, b in bad.head(15).iterrows():
            print(f"  [{b['qid']}] {b['sentence'][:90]}  ← {b['judge_reason'][:60]}")


CORRUPT_SCHEMA = {
    "type": "object",
    "properties": {"corrupted": {"type": "string"}, "what_changed": {"type": "string"}},
    "required": ["corrupted", "what_changed"],
    "additionalProperties": False,
}
CORRUPT_PROMPT = """다음 문장에 **사실 오류를 정확히 하나** 넣어 다시 쓰세요.
숫자·연도·기관명·국가·주체를 바꾸거나, 긍정/부정을 뒤집거나, 원문에 없는 주장을 하나 덧붙이는 방식 중 하나.
문체와 길이, 끝의 인용 표기([n])는 그대로 유지하세요. 오류가 눈에 띄게 티 나지 않도록 자연스럽게.

문장: {sentence}"""


def negative_control(n=15, seed=11):
    """
    채점관 민감도 검증: 실제로 '근거 있음' 판정된 문장에 사실 오류를 하나 주입 → 채점관이 잡아내는가?
    - detection_rate: 오류 주입 문장을 unsupported로 판정한 비율 (높을수록 좋음)
    - 원문 문장의 supported 비율과 함께 보면 채점관의 판별력을 알 수 있음
    """
    from student.common.llm import claude_json
    from student.common.schemas import Day2Plan
    from student.day2.impl.embeddings import Embeddings
    from student.day2.impl.rag import Day2Agent

    sdf = pd.read_csv(RESULT_DIR / "faithfulness_sentences.csv")
    ans = pd.read_csv(RESULT_DIR / "faithfulness_answers.csv").set_index("id")
    cand = sdf[sdf["judge_supported"] & (sdf["sentence"].str.len() > 25)]
    pick = cand.groupby("qid").head(1).sample(n, random_state=seed)

    plan = Day2Plan(); agent = Day2Agent(plan); emb = Embeddings(model=plan.embedding_model)
    corrupt_cache, judge_cache = JsonlCache("negative_control_corrupt"), JsonlCache("negative_control_judge")
    rows = []
    for _, r in pick.iterrows():
        q = ans.loc[r["qid"], "query"]
        contexts, _ = agent.retrieve(q, plan, emb)
        ctx_text = "\n\n".join(f"[{i}] {re.sub(r'\s+', ' ', c['chunk'])}" for i, c in enumerate(contexts, 1))
        c = corrupt_cache.get_or_call({"v": 1, "s": r["sentence"]},
                                      lambda: claude_json(CORRUPT_PROMPT.format(sentence=r["sentence"]), CORRUPT_SCHEMA, max_tokens=800))
        prompt = JUDGE_PROMPT.format(context=ctx_text, question=q, evidence_block="",
                                     sentences=f"0. {c['corrupted']}", evidence_rule="")
        j = judge_cache.get_or_call({"v": 1, "q": q, "s": c["corrupted"]},
                                    lambda: claude_json(prompt, JUDGE_SCHEMA, max_tokens=1500))
        flagged = not j["sentences"][0]["supported"] if j["sentences"] else False
        rows.append({"qid": r["qid"], "sent_idx": r["sent_idx"], "query": q, "original": r["sentence"],
                     "corrupted": c["corrupted"], "what_changed": c["what_changed"],
                     "judge_flagged_corrupted": flagged,
                     "judge_reason": j["sentences"][0]["reason"] if j["sentences"] else ""})
    print(corrupt_cache.stats(), judge_cache.stats())
    df = save_csv(rows, "faithfulness_negative_control.csv")
    rate = df["judge_flagged_corrupted"].mean()
    save_csv([{"n": len(df), "detection_rate": round(rate, 3)}], "faithfulness_negative_control_summary.csv")
    print(f"오류 주입 문장 {len(df)}개 중 채점관이 잡아낸 비율: {rate:.3f}")
    for _, r in df[~df["judge_flagged_corrupted"]].iterrows():
        print(f"  놓침 [{r['qid']}] {r['what_changed'][:50]} :: {r['corrupted'][:80]}")


def export_human(n_each=10, seed=7):
    """사람 라벨용: 원문 문장 10 + 오류 주입 문장 10 을 섞어서(정답 숨김) 출제"""
    nc = pd.read_csv(RESULT_DIR / "faithfulness_negative_control.csv")
    sdf = pd.read_csv(RESULT_DIR / "faithfulness_sentences.csv")
    ans = pd.read_csv(RESULT_DIR / "faithfulness_answers.csv").set_index("id")
    bad = nc.sample(min(n_each, len(nc)), random_state=seed)
    good = sdf[~sdf["qid"].isin(bad["qid"])].sample(n_each, random_state=seed)
    items = [{"item": f"b{i}", "qid": r["qid"], "query": r["query"], "sentence": r["corrupted"],
              "judge_supported": not r["judge_flagged_corrupted"], "is_corrupted": True}
             for i, (_, r) in enumerate(bad.iterrows())]
    items += [{"item": f"g{i}", "qid": r["qid"], "query": ans.loc[r["qid"], "query"], "sentence": r["sentence"],
               "judge_supported": bool(r["judge_supported"]), "is_corrupted": False}
              for i, (_, r) in enumerate(good.iterrows())]
    df = pd.DataFrame(items).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["no"] = range(1, len(df) + 1)
    df.to_csv(RESULT_DIR / "faithfulness_human_key.csv", index=False, encoding="utf-8-sig")   # 정답키(보지 말 것)
    df[["no", "qid", "query", "sentence"]].assign(human_supported="").to_csv(HUMAN_LABEL_FILE, index=False, encoding="utf-8-sig")
    print(f"→ {HUMAN_LABEL_FILE}: human_supported 열에 1(문서 근거 있음)/0(없음 또는 틀림) 입력")
    print("  각 질문의 검색 근거는 eval/results/faithfulness_contexts.md 에서 확인")
    # 사람이 판단할 수 있도록 질문별 컨텍스트 덤프
    from student.common.schemas import Day2Plan
    from student.day2.impl.embeddings import Embeddings
    from student.day2.impl.rag import Day2Agent
    plan = Day2Plan(); agent = Day2Agent(plan); emb = Embeddings(model=plan.embedding_model)
    qcache = JsonlCache("query_embeddings")
    lines = ["# 사람 라벨링용 근거 문서\n"]
    for qid in dict.fromkeys(df["qid"]):
        q = ans.loc[qid, "query"]
        contexts, _ = agent.retrieve(q, plan, emb)
        lines.append(f"## {qid} — {q}\n")
        lines += [f"**[{i}]** {re.sub(r'\s+', ' ', c['chunk'])}\n" for i, c in enumerate(contexts, 1)]
    (RESULT_DIR / "faithfulness_contexts.md").write_text("\n".join(lines), encoding="utf-8")


def kappa():
    h = pd.read_csv(HUMAN_LABEL_FILE)
    h = h[h["human_supported"].astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
    key = pd.read_csv(RESULT_DIR / "faithfulness_human_key.csv")
    m = h[["no", "human_supported"]].merge(key, on="no")
    hum = m["human_supported"].astype(float).astype(int).astype(bool).tolist()
    jud = m["judge_supported"].astype(bool).tolist()
    truth = (~m["is_corrupted"].astype(bool)).tolist()
    agree = sum(a == b for a, b in zip(hum, jud)) / len(m)
    k = cohen_kappa(hum, jud)
    res = {"n": len(m), "judge_vs_human_agreement": round(agree, 3), "cohen_kappa": round(k, 3),
           "human_accuracy_vs_injected": round(sum(a == b for a, b in zip(hum, truth)) / len(m), 3),
           "judge_accuracy_vs_injected": round(sum(a == b for a, b in zip(jud, truth)) / len(m), 3)}
    save_csv([res], "faithfulness_kappa.csv")
    print(res)
    for _, r in m[[a != b for a, b in zip(hum, jud)]].iterrows():
        print(f"  불일치 #{r['no']} 사람={int(float(r['human_supported']))} 채점관={int(r['judge_supported'])} :: {r['sentence'][:80]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-human", action="store_true")
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--negative-control", action="store_true")
    a = ap.parse_args()
    if a.negative_control:
        negative_control()
    elif a.export_human:
        export_human()
    elif a.kappa:
        kappa()
    else:
        run_pipeline()
