# -*- coding: utf-8 -*-
"""
RAG 검색 평가 + 청킹/하이브리드 실험

실험 설정 (--exp):
  A: 베이스라인  chunk 1200 / overlap 200, 벡터 검색
  B: chunk 500 / overlap 100, 벡터 검색
  C: B 인덱스 + 문자 bigram BM25 결합(RRF)  ※ 한국어 PDF는 띄어쓰기가 깨져 있어 단어 대신 문자 n-gram 사용
  D: A 인덱스 + 하이브리드 (청크 크기 효과와 하이브리드 효과를 분리하기 위한 대조군)

정답 기준: 질문마다 '근거 구절(evidence)'을 저장해 두고, 검색된 청크에 그 구절이 (공백 무시) 포함되면 정답.
           → 청크 경계가 달라져도 같은 정답셋으로 비교 가능 (overlap ≥ 구절 길이라 구절이 잘리지 않음)
지표: Hit@1/3/5, MRR (답변 가능 질문 기준)
부산물: 질문별 top_score / mean_top5 → 신뢰도 게이트 임계값 분석(run_gate.py)에 사용

사용법: python -m eval.run_retrieval --exp A
"""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from eval.common import DATA_DIR, ROOT, JsonlCache, load_jsonl, save_csv
from eval.metrics import hit_at_k, mrr
from student.day2.impl.lexical import CharBM25, rrf_merge

EXPS = {
    "A": {"chunk_size": 1200, "chunk_overlap": 200, "index": "exp_A", "hybrid": False},
    "B": {"chunk_size": 500, "chunk_overlap": 100, "index": "exp_B", "hybrid": False},
    "C": {"chunk_size": 500, "chunk_overlap": 100, "index": "exp_B", "hybrid": True},
    "D": {"chunk_size": 1200, "chunk_overlap": 200, "index": "exp_A", "hybrid": True},
}
TOP_K = 5
EMBED_MODEL = "text-embedding-3-small"


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def ensure_index(cfg) -> Path:
    from student.day2.impl.build_index import build_index
    d = ROOT / "indices" / cfg["index"]
    if not (d / "faiss.index").exists():
        print(f"[index] building {d} (chunk {cfg['chunk_size']}/{cfg['chunk_overlap']}) ...")
        build_index(["data/raw"], str(d), model=EMBED_MODEL,
                    chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="A", choices=list(EXPS))
    ap.add_argument("--data", default="rag_qa.jsonl")
    ap.add_argument("--tag", default="", help="결과 파일 접미사 (예: human)")
    args = ap.parse_args()
    cfg = EXPS[args.exp]
    name = f"{args.exp}_{args.tag}" if args.tag else args.exp

    from student.day2.impl.store import FaissStore
    from student.day2.impl.embeddings import Embeddings

    data = load_jsonl(DATA_DIR / args.data)
    idx = ensure_index(cfg)
    store = FaissStore.load(str(idx / "faiss.index"), str(idx / "docs.jsonl"))
    texts = [d["text"] for d in store.docs]
    bm25 = CharBM25(texts) if cfg["hybrid"] else None

    emb = Embeddings(model=EMBED_MODEL)
    cache = JsonlCache("query_embeddings")
    rows = []
    for d in data:
        vec = np.array(cache.get_or_call({"model": EMBED_MODEL, "q": d["query"]},
                                         lambda: emb.encode([d["query"]])[0].tolist()), dtype="float32")
        dense = store.search(vec, top_k=len(texts))           # 전체 순위 (코퍼스가 작아 가능)
        dense_ids = [c["doc_id"] for c in dense]
        dense_scores = [c["score"] for c in dense]
        if bm25 is not None:
            lex_order = np.argsort(-bm25.scores(d["query"]))
            lex_ids = [store.docs[i]["id"] for i in lex_order]
            ranked_ids = rrf_merge(dense_ids, lex_ids)
        else:
            ranked_ids = dense_ids

        rank = None
        if d["answerable"] and d.get("evidence"):
            ev = _norm(d["evidence"])
            id2text = {x["id"]: x["text"] for x in store.docs}
            for r, cid in enumerate(ranked_ids, 1):
                if ev in _norm(id2text[cid]):
                    rank = r
                    break
        rows.append({
            "id": d["id"], "query": d["query"], "answerable": d["answerable"],
            "has_evidence": bool(d.get("evidence")), "rank": rank,
            "top_score": dense_scores[0], "mean_top5": float(np.mean(dense_scores[:TOP_K])),
            "top1_chunk": ranked_ids[0],
        })
    print(cache.stats())

    df = save_csv(rows, f"retrieval_{name}.csv")
    ans = df[df["answerable"] & df["has_evidence"]]
    if ans.empty:
        print("(근거 구절이 없는 데이터셋 → Hit@k 생략, 게이트 분석용 점수만 저장)")
        return
    ranks = [None if (isinstance(r, float) and math.isnan(r)) or r is None else int(r) for r in ans["rank"]]
    summary = {"exp": args.exp, **{k: cfg[k] for k in ("chunk_size", "chunk_overlap", "hybrid")},
               "n_chunks": len(texts), "n_answerable": len(ans),
               "hit@1": round(hit_at_k(ranks, 1), 3), "hit@3": round(hit_at_k(ranks, 3), 3),
               "hit@5": round(hit_at_k(ranks, 5), 3), "mrr": round(mrr(ranks), 3)}
    save_csv([summary], f"retrieval_{name}_summary.csv")
    print(summary)
    miss = ans[[r is None or r > TOP_K for r in ranks]]
    if len(miss):
        print(f"Top-{TOP_K} 밖 {len(miss)}건:")
        for _, m in miss.iterrows():
            print(f"  [{m['id']}] rank={m['rank']} {m['query']}")


if __name__ == "__main__":
    main()
