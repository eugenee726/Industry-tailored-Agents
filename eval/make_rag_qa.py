# -*- coding: utf-8 -*-
"""
RAG 평가용 질문 초안 생성 → 사람이 검수하는 용도

- 답변 가능 질문: 베이스라인 코퍼스(indices/day2/docs.jsonl)에서 본문 청크 40개를 고르게 샘플링,
  Claude가 청크마다 질문 1개 + 그 답의 근거가 되는 '원문 그대로의 구절(evidence)'을 생성.
  evidence는 청크 경계와 무관한 정답 기준이 된다(청크 크기 실험에서도 같은 정답 사용).
- 검증: evidence가 실제 청크에 (공백 무시) 포함되지 않으면 버린다.
- 답변 불가 질문 20건: 코퍼스 밖 주제로 사람이 작성 (아래 OUT_OF_SCOPE).

출력: eval/datasets/rag_qa.draft.jsonl  → 검수 후 rag_qa.jsonl 로 저장
"""
from __future__ import annotations

import json
import re

from eval.common import DATA_DIR, ROOT, JsonlCache, save_jsonl
from student.common.llm import claude_json

N_ANSWERABLE = 40

SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["question", "evidence"],
    "additionalProperties": False,
}

PROMPT = """다음은 AI 의료기기 규제 관련 한국어 문서의 일부입니다 (PDF 추출이라 띄어쓰기가 깨져 있을 수 있음).

<chunk>
{chunk}
</chunk>

이 구간만 읽으면 답할 수 있는 질문을 하나 만드세요.
- 실무자가 실제로 물어볼 법한 자연스러운 한국어 질문 (띄어쓰기 정상)
- "이 문서에서", "위 내용에서" 같은 표현은 쓰지 마세요
- 질문에 정답 문구를 그대로 넣지 마세요
- evidence: 답의 근거가 되는 구절을 chunk에서 **글자 그대로** 20~80자 복사 (띄어쓰기도 원문 그대로)
- 서지/참고문헌만 있는 구간이면 question을 "SKIP"으로 하세요"""

OUT_OF_SCOPE = [
    "삼성전자 2026년 3분기 영업이익은 얼마야?",
    "NIPA AI 바우처 사업의 올해 접수 마감일은?",
    "엔비디아 최신 GPU 모델 이름이 뭐야?",
    "서울 강남구 맛집 추천해줘",
    "파이썬에서 리스트를 정렬하는 방법은?",
    "2026년 최저임금은 얼마야?",
    "루닛의 작년 매출액은?",
    "중소기업 R&D 바우처 신청 자격 요건",
    "EU AI Act 시행 일정은 어떻게 돼?",
    "의료기기 회사 채용 공고 있어?",
    "원격진료 수가는 얼마로 책정됐어?",
    "식약처장 이름이 뭐야?",
    "디지털헬스케어 스타트업 투자 유치 순위",
    "병원 전자의무기록(EMR) 시스템 도입 비용",
    "ChatGPT 의료 상담 정확도 연구 결과",
    "개인정보보호법상 가명정보 처리 절차",
    "2026년 건강보험료율 인상폭",
    "한국 반도체 수출 실적 추이",
    "의료 AI 관련 국내 특허 출원 건수",
    "애플워치 심전도 기능 FDA 승인 시기",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _is_body_chunk(text: str) -> bool:
    """본문 청크만 사용: 한글 비율이 낮거나(참고문헌/영문 초록) 너무 짧으면 제외"""
    hangul = len(re.findall(r"[가-힣]", text))
    return len(text) > 400 and hangul / max(1, len(text)) > 0.35


def main():
    docs = [json.loads(l) for l in open(ROOT / "indices/day2/docs.jsonl", encoding="utf-8")]
    body = [d for d in docs if _is_body_chunk(d["text"])]
    step = max(1, len(body) // (N_ANSWERABLE + 8))      # 여유분 포함 균등 샘플
    picked = body[::step]
    cache = JsonlCache("rag_qa_gen")
    print(f"[rag_qa] 본문 청크 {len(body)}/{len(docs)}개 중 {len(picked)}개 후보 (API 최대 {len(picked)}회)")

    rows, dropped = [], 0
    for d in picked:
        if len([r for r in rows if r["answerable"]]) >= N_ANSWERABLE:
            break
        out = cache.get_or_call({"chunk_id": d["id"], "v": 1},
                                lambda: claude_json(PROMPT.format(chunk=d["text"]), SCHEMA, max_tokens=1024))
        q, ev = out["question"].strip(), out["evidence"].strip()
        if q == "SKIP" or len(_norm(ev)) < 10 or _norm(ev) not in _norm(d["text"]):
            dropped += 1
            continue
        rows.append({"id": f"q{len(rows) + 1:02d}", "query": q, "answerable": True,
                     "evidence": ev, "source_chunk": d["id"], "reviewed": False})
    for i, q in enumerate(OUT_OF_SCOPE, 1):
        rows.append({"id": f"o{i:02d}", "query": q, "answerable": False, "evidence": "",
                     "source_chunk": "", "reviewed": True})

    out_path = DATA_DIR / "rag_qa.draft.jsonl"
    save_jsonl(out_path, rows)
    print(cache.stats())
    print(f"답변 가능 {sum(r['answerable'] for r in rows)}건 (검증 탈락 {dropped}건), 답변 불가 {len(OUT_OF_SCOPE)}건 → {out_path}")


if __name__ == "__main__":
    main()
