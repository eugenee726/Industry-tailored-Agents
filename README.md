# Industry-tailored Agents

> **산업 맞춤형 정보 수집·분석 멀티에이전트** — Google ADK 오케스트레이터가 질의를 웹 리서치 / 문서 RAG / 정부 공고 에이전트로 라우팅합니다.
> 이 저장소의 핵심은 기능 자체보다 **"LLM 결과를 지표로 정의하고, 측정하고, 개선한 과정"** 입니다. → [평가 리포트](eval/REPORT.md)

---

## 평가 & 개선 결과 요약

모든 수치는 `eval/`의 스크립트로 재현됩니다. 결과 CSV와 API 응답 캐시가 포함되어 있어 LLM 재호출 비용은 0입니다. 검색 인덱스는 원문 저작권 때문에 빼 두었으므로 [출처](#rag-원문-출처)의 PDF로 다시 만들어야 합니다(임베딩 비용 몇 센트).

| 영역 | 무엇을 쟀나 | 개선 전 | 개선 후 | 무엇을 바꿨나 |
|---|---|---:|---:|---|
| 라우팅 | 정확도 · 홀드아웃 24건 | 91.7% | **100%** | v1에서 틀리거나 흔들린 8건이 모두 "문서 주제인데 '문서' 키워드가 없는 질의" → 프롬프트에 인덱스 문서 주제 명시, 직접 답변 금지 |
| 라우팅 | 3회 반복 일관성 · 홀드아웃 | 91.7% | **100%** | 〃 |
| RAG 검색 | MRR · 40건 | 0.632 | **0.844** | 벡터 + 문자 bigram BM25 하이브리드(RRF), 500자 청크. 대조군으로 하이브리드 효과(+0.18)와 청크 효과 분리 |
| RAG 검색 | Hit@5 | 0.90 | **1.00** | 〃 |
| 신뢰도 게이트 | 코퍼스 밖 질문 22건 중 통과(=근거 없는 답변 위험) | 13건 (README 0.32) | **3건 → 생성 단계에서 전부 거절** | 88건으로 임계값 스윕, 3구간 정책(답변 / 경고 / 거절) |
| 답변 생성 | 답 없는 질문 거절 정확도 | 생성 단계 없음 | **100%** (22/22) | Claude 생성 + 문장별 [n] 인용 강제, 근거 없으면 정해진 문구로 거절 |
| 답변 생성 | 인용 유효율 / Faithfulness(LLM 채점) | — | **100% / 100%** | 채점관 검증: 사실 오류를 하나씩 주입한 문장 15개를 **15개 모두 탐지**. 사람 블라인드 라벨(7문장, 진행 중)과의 불일치 2건은 모두 사람이 놓친 오류 주입 문장 |
| Day1 | 티커 추출 정밀도 · 30건 | 0.37 | **0.94** | `"ai 의학"` → `AI` 티커 오인식 등 수정 |
| Day3 | 공고 마감일 채움률 · 20건 | 0% | **100%** | 마감일이 항상 비어 랭킹 가중치 50%가 무효였던 문제 수정 |

**검증 과정에서 발견한 숨은 버그**
- Day1 "LLM 기업 요약"이 존재하지 않는 메서드(`LiteLlm.invoke`)를 호출해 **한 번도 동작하지 않았음** (예외를 삼켜 빈 문자열 반환) → 수정
- RAG 게이트 임계값이 README(0.32)와 코드(0.20)에서 달랐고, 둘 다 코퍼스 밖 질문을 다수 통과시킴 → 측정 기반으로 재설정
- "RAG"라고 했지만 생성 단계 없이 검색 청크를 이어 붙이기만 했음 → 근거 인용 생성 추가

자세한 방법·혼동행렬·한계는 [eval/REPORT.md](eval/REPORT.md).

---

## 시스템 구조

```
사용자 질의
     │
     ▼
Industry_Tailored_Orchestrator  (Claude Sonnet 5.5, 라우팅 전용 — 직접 답변 금지)
     │
     ├── Day1WebAgent   웹 검색(Tavily) + 주가(yfinance) + 기업 개요 요약(Claude)
     ├── Day2RagAgent   하이브리드 검색 → 신뢰도 게이트 → 근거 인용 답변(Claude)
     └── Day3GovAgent   정부 공고 수집(NIPA·기업마당·웹) → 정규화(마감일 추출) → 랭킹
```

### Day2 RAG 파이프라인

```
질의 ─▶ 벡터 검색(OpenAI 임베딩, FAISS) ─┐
     └▶ 문자 bigram BM25 ───────────────┴▶ RRF 결합 → Top-5
                                              │
                    dense top_score ──▶ 게이트 ┤ ≥0.54 답변 / 0.45~0.54 경고+답변 / <0.45 거절
                                              ▼
                         Claude: 컨텍스트만 사용, 문장마다 [n] 인용, 근거 없으면 거절 문구
```

- 한국어 PDF 추출 텍스트는 띄어쓰기가 깨져 있어(`식약처는2025년5월7일자로`) 단어 대신 **문자 bigram**으로 BM25 구성
- 게이트와 생성 단계의 **2중 방어**: 게이트 '경고' 구간을 통과한 코퍼스 밖 질문 3건은 생성 단계에서 모두 거절됨

### Day3 공고 랭킹

`최종 점수 = 마감 임박도 50% + 키워드 일치 30% + 출처 신뢰도 20%` (+ 공공 도메인 가점, 목록 URL 감점)
마감일은 제목·스니펫에서 규칙 기반 추출 (`접수마감` 명시 > 기간 표현의 끝 날짜 > `~까지` > 연도 없는 표현)

---

## 평가 체계 (`eval/`)

| 스크립트 | 측정 대상 | 데이터 |
|---|---|---|
| `run_routing.py` | 라우팅 정확도·클래스별 P/R·혼동행렬·반복 일관성 | 개발셋 60 + 홀드아웃 24 |
| `run_retrieval.py` | Hit@1/3/5, MRR (청크·하이브리드 실험 A~D) | 근거 구절이 달린 질문 40 + 코퍼스 밖 20 |
| `run_gate.py` | 임계값별 Precision/Recall, 3구간 정책 | 위 60 + 사람이 쓴 문서 질문 28 |
| `run_faithfulness.py` | 답변률, 거절 정확도, 인용 유효율, 문장별 Faithfulness, 채점관 검증 | 88 |
| `run_tickers.py`, `run_close_date.py` | 규칙 기반 컴포넌트 수정 전후 | 30 / 20 |
| `make_report.py` | 위 결과 → `REPORT.md` + 차트 | — |

설계 원칙
- **실제 시스템을 그대로 측정**: 라우팅은 ADK 오케스트레이터를 실제로 실행(서브에이전트만 스텁), RAG는 운영 코드 경로 그대로
- **과적합 경계**: 프롬프트 개선 후 처음 보는 홀드아웃으로 재평가 / LLM 생성 질문만으로 임계값을 정하면 높게 잡히는 문제를 사람이 쓴 질문으로 보정
- **채점관도 검증 대상**: LLM-as-judge 결과를 음성 대조군(오류 주입 문장 탐지율)과 사람 라벨 일치도(Cohen's κ)로 확인
- **재현성·비용**: 모든 API 응답을 `eval/cache/`에 저장. 전체 평가 실행 비용은 몇 달러 수준(추정)

```bash
python -m eval.run_routing --prompt v2 --data routing_holdout
python -m eval.run_retrieval --exp C
python -m eval.run_gate --exp C
python -m eval.run_faithfulness
python -m eval.make_report
```

---

## 기술 스택

| 분류 | 기술 |
|---|---|
| Agent Framework | Google ADK 1.28.0 (LiteLLM 경유) |
| LLM | Claude Sonnet 5.5 (`claude-sonnet-5-5`) — 오케스트레이터, Day1 요약, Day2 생성, 평가 채점 |
| 임베딩 / 벡터 검색 | OpenAI `text-embedding-3-small` / FAISS (IndexFlatIP) |
| 키워드 검색 | 문자 bigram BM25 (직접 구현, 의존성 없음) |
| 웹 검색 / 주가 | Tavily Search API / yfinance |
| 평가·리포트 | pandas, matplotlib |

---

## 프로젝트 구조

```
Industry-tailored-Agents/
├── apps/root_app/          # 오케스트레이터 (agent.py, prompts.py: v1/v2)
├── student/                # 서브에이전트 (교육 과정 미니프로젝트의 Day1~3 구조 유지)
│   ├── common/             # 스키마, 렌더러, 공통 LLM 설정(llm.py)
│   ├── day1/               # 웹 리서치
│   ├── day2/impl/          # RAG: ingest, embeddings, store(FAISS), lexical(BM25/RRF), rag
│   └── day3/impl/          # 공고: fetchers, normalize(마감일 추출), rank
├── eval/                   # 평가 스크립트, 데이터셋, 결과, 캐시, REPORT.md
├── indices/                # (저장소 미포함) day2: 운영 인덱스, exp_A·exp_B: 실험용. PDF로 재구축
└── data/raw/               # RAG 원문 PDF (저작권 문제로 저장소에 미포함, 아래 출처에서 받아 넣기)
```

### RAG 원문 출처
저작권이 있는 문서라 PDF는 올리지 않았습니다. 인덱스 재구축이 필요하면 아래 문서를 받아 `data/raw/`에 넣으세요.
1. 박혜진, 「의료 인공지능의 활용을 둘러싼 법적 과제: 규제의 진화 및 책임의 배분을 중심으로」, 『비교사법』 제29권 4호, 한국비교사법학회, 2022, 217–251면. https://doi.org/10.22922/jcpl.2022.29.4.217
2. 법무법인 린(LIN), 「AI 의료기기를 둘러싼 주요 쟁점과 보험시장의 변화」, *Tech Legal Insights* AID Vol. 6, 2025.07. https://www.law-lin.com

---

## 실행

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
```

`.env`
```env
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...        # 임베딩
TAVILY_API_KEY=...        # 웹 검색
```

```bash
# RAG 인덱스 구축 (최초 1회) — data/raw/에 위 출처의 PDF를 넣은 뒤. 운영 설정: 500자 청크 / 100자 겹침
# 실험용 인덱스(exp_A, exp_B)는 run_retrieval.py 실행 시 없으면 자동 생성
python -c "from student.day2.impl.build_index import build_index; build_index(['data/raw'], 'indices/day2', chunk_size=500, chunk_overlap=100)"

adk web apps              # http://localhost:8000
```

---

## 기여 범위

- **1차 (2025.11, KT AIVLE 팀 프로젝트, 기여도 20%)**: 교육 과정이 제공한 Day1~3 에이전트 뼈대(`student/`) 위에서 에이전트 구조·코드 수정 담당
- **2차 (2026.11, 개인)**: 품질 평가 체계 구축과 개선
  - 평가 체계 전체(`eval/`): 데이터셋 설계, 지표 정의, 실험, 리포트 자동화
  - 오케스트레이터 프롬프트 v2, RAG 하이브리드 검색·3구간 게이트·근거 인용 생성
  - 버그 수정: Day1 요약 미동작, 티커 오인식, Day3 마감일 미추출
  - LLM을 Claude Sonnet 5.5로 전환하고 모델 설정을 `student/common/llm.py`로 일원화
