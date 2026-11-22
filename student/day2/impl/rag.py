# -*- coding: utf-8 -*-
"""
Day2 RAG: 검색 → 신뢰도 게이트 → 근거 인용 답변 생성

설계 근거는 eval/ 의 측정 결과 (eval/REPORT.md):
  - 검색: 500자 청크 + 벡터/문자-BM25 하이브리드(RRF)  … run_retrieval 실험 C
  - 게이트: dense top_score 기준 3구간                    … run_gate
        top ≥ confident_score(0.54) → answer   (평가셋에서 '답 없는 질문' 통과 0건)
        top ≥ min_score(0.45)       → warn     (답변 + '근거 약함' 경고)
        그 외                        → refuse   (생성하지 않음)
  - 생성: Claude가 컨텍스트만 사용해 문장마다 [n] 인용, 근거 없으면 정해진 문구로 거절 … run_faithfulness
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

from student.common.schemas import Day2Plan
from .embeddings import Embeddings
from .lexical import CharBM25, rrf_merge
from .store import FaissStore

NO_EVIDENCE = "문서에서 근거를 찾지 못했습니다."

GEN_SYSTEM = """당신은 보유 문서만을 근거로 답하는 리서치 어시스턴트입니다.
규칙:
1) 아래 <context>의 내용만 사용하세요. 일반 지식이나 추측으로 내용을 보태지 마세요.
2) 답변의 모든 문장 끝에 근거가 된 컨텍스트 번호를 [1], [2]처럼 붙이세요. 여러 개면 [1][3].
3) 컨텍스트에 질문의 답이 없으면 다른 말 없이 정확히 "문서에서 근거를 찾지 못했습니다." 한 문장만 출력하세요.
4) 한국어로 3~6문장 이내로 간결하게 답하세요. 원문이 띄어쓰기가 깨져 있어도 답변은 정상 띄어쓰기로 쓰세요."""


def _idx_paths(index_dir: str):
    return os.path.join(index_dir, "faiss.index"), os.path.join(index_dir, "docs.jsonl")


def _load_store(plan: Day2Plan, emb: Embeddings) -> FaissStore:
    index_path, docs_path = _idx_paths(plan.index_dir)
    if not (os.path.exists(index_path) and os.path.exists(docs_path)):
        raise FileNotFoundError(f"FAISS 인덱스가 없습니다. 먼저 ingest를 실행하세요: {plan.index_dir}")
    return FaissStore.load(index_path, docs_path)


def _gate(dense_scores: List[float], plan: Day2Plan) -> Dict[str, Any]:
    if not dense_scores:
        return {"status": "insufficient", "zone": "refuse", "top_score": 0.0, "mean_topk": 0.0}
    top = float(dense_scores[0])
    mean_topk = float(np.mean(dense_scores[:plan.top_k]))
    if top >= plan.confident_score and mean_topk >= plan.min_mean_topk:
        zone = "answer"
    elif top >= plan.min_score and mean_topk >= plan.min_mean_topk:
        zone = "warn"
    else:
        zone = "refuse"
    return {"status": "insufficient" if zone == "refuse" else "enough", "zone": zone,
            "top_score": top, "mean_topk": mean_topk}


def _draft_answer(query: str, contexts: List[Dict[str, Any]], plan: Day2Plan) -> str:
    """LLM 호출 실패 시 폴백: 근거 청크 발췌"""
    buf, budget = [], plan.max_context
    for c in contexts:
        t = c["chunk"].strip().replace("\n", " ")
        if len(t) > budget:
            t = t[:budget] + "..."
        buf.append(f"- {t}")
        budget -= len(t)
        if budget <= 0:
            break
    return f"질의: {query}\n\n핵심 근거 발췌:\n" + "\n".join(buf) if buf else ""


def build_generation_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    ctx = "\n\n".join(f"[{i}] {re.sub(r'\s+', ' ', c['chunk']).strip()}" for i, c in enumerate(contexts, 1))
    return f"<context>\n{ctx}\n</context>\n\n질문: {query}"


def generate_answer(query: str, contexts: List[Dict[str, Any]]) -> str:
    from student.common.llm import claude_text
    return claude_text(build_generation_prompt(query, contexts), system=GEN_SYSTEM, max_tokens=1500, effort="low")


class Day2Agent:
    def __init__(self, plan_defaults: Day2Plan = Day2Plan()):
        self.plan_defaults = plan_defaults
        self._store: Optional[FaissStore] = None
        self._bm25: Optional[CharBM25] = None
        self._loaded_dir: Optional[str] = None

    def _ensure_loaded(self, plan: Day2Plan, emb: Embeddings):
        if self._loaded_dir != plan.index_dir:
            self._store = _load_store(plan, emb)
            self._bm25 = CharBM25([d["text"] for d in self._store.docs])
            self._loaded_dir = plan.index_dir

    def retrieve(self, query: str, plan: Day2Plan, emb: Embeddings):
        """반환: (상위 K 컨텍스트, 전체 dense 점수 내림차순)"""
        self._ensure_loaded(plan, emb)
        store = self._store
        qv = emb.encode([query])[0]
        dense = store.search(qv, top_k=len(store.docs))
        dense_scores = [c["score"] for c in dense]
        if plan.hybrid:
            by_id = {c["doc_id"]: c for c in dense}
            lex_order = np.argsort(-self._bm25.scores(query))
            ranked = rrf_merge([c["doc_id"] for c in dense], [store.docs[i]["id"] for i in lex_order])
            contexts = [by_id[i] for i in ranked[:plan.top_k]]
        else:
            contexts = dense[:plan.top_k]
        return contexts, dense_scores

    def handle(self, query: str, plan: Day2Plan = None) -> Dict[str, Any]:
        plan = plan or self.plan_defaults
        emb = Embeddings(model=plan.embedding_model)
        contexts, dense_scores = self.retrieve(query, plan, emb)
        gate = _gate(dense_scores, plan)

        payload: Dict[str, Any] = {
            "type": "rag_answer",
            "query": query,
            "plan": plan.__dict__,
            "contexts": contexts,
            "gating": gate,
            "answer": "",
            "answer_source": "",
        }
        if plan.force_rag_only or (gate["zone"] != "refuse" and plan.return_draft_when_enough):
            if plan.generate:
                try:
                    payload["answer"] = generate_answer(query, contexts)
                    payload["answer_source"] = "llm"
                except Exception as e:  # LLM 실패 시 발췌로 폴백
                    payload["answer"] = _draft_answer(query, contexts, plan)
                    payload["answer_source"] = f"excerpt_fallback ({type(e).__name__})"
            else:
                payload["answer"] = _draft_answer(query, contexts, plan)
                payload["answer_source"] = "excerpt"
        return payload
