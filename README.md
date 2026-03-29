# Industry-tailored Agents

> **산업 맞춤형 정보 수집·분석 자동화 에이전트 시스템**
> Google ADK 기반 Multi-Agent 아키텍처로 웹 검색, 문서 RAG, 정부 공고 수집을 자동화합니다.

---

## 프로젝트 개요

기업 실무자가 산업 동향, 내부 문서, 정부 지원사업 공고를 하나의 인터페이스에서 조회할 수 있도록
**오케스트레이터(Orchestrator) + 3개의 전문 서브 에이전트**로 구성된 자동화 시스템입니다.

사용자의 자연어 질의를 분석하여 적합한 에이전트로 자동 라우팅하고,
수집된 데이터를 정규화·스코어링한 뒤 마크다운 리포트로 자동 생성합니다.

---

## 시스템 아키텍처

```
사용자 질의
     │
     ▼
Industry_Tailored_Orchestrator   ← 질의 분석 & 라우팅
     │
     ├── Day1WebAgent    웹 검색 + 주가 + 기업 개요
     ├── Day2RagAgent    로컬 문서 RAG (FAISS 유사도 검색)
     └── Day3GovAgent    정부/공공 공고 수집 & 랭킹
```

### 라우팅 기준

| 질의 유형 | 담당 에이전트 |
|---|---|
| 최신 뉴스 / 주가 / 기업 동향 | `Day1WebAgent` |
| 내부 문서 요약 / 근거 검색 | `Day2RagAgent` |
| 정부 공고 / 바우처 / 지원사업 | `Day3GovAgent` |
| RAG 결과 신뢰도 낮음 | `Day2 → Day1` 순차 호출 (Web fallback) |

---

## 에이전트별 상세

### Day1 — 웹 리서치 에이전트

- Tavily Search API로 최신 웹 정보 수집
- `yfinance` 기반 실시간 주가 스냅샷 (미국/한국 티커 자동 인식)
- 기업 개요 추출 후 LLM으로 3~5문장 요약
- 티커 자동 추출: 영문 1~5자(`AAPL`, `NVDA`) + 국내 6자리 코드(`005930.KS`)

### Day2 — RAG 에이전트

- FAISS 벡터 인덱스 기반 유사도 검색 (`text-embedding-3-small`)
- 신뢰도 게이트: `top_score ≥ 0.32`, `mean_topk ≥ 0.30` 미달 시 "근거 불충분" 처리
- Top-K 근거 문서를 score와 함께 표로 출력
- 인덱스 사전 구축 필요: `python -m student.day2.impl.ingest`

### Day3 — 정부 공고 에이전트

**데이터 파이프라인: 수집 → 정규화 → 랭킹**

```
fetchers.py   →   normalize.py   →   rank.py
  (수집)            (정규화)           (스코어링)
```

**수집 소스**

| 소스 | 방식 |
|---|---|
| NIPA (nipa.kr) | 도메인 한정 검색 |
| 기업마당 (bizinfo.go.kr) | 도메인 한정 검색 |
| 일반 웹 | Fallback 검색 |

**스코어링 모델**

```
최종 점수 = 마감 임박도(50%) + 키워드 일치도(30%) + 출처 신뢰도(20%)
```

| 요소 | 가중치 | 계산 방식 |
|---|---:|---|
| 마감 임박도 | 50% | 30일 이내 선형 증가, 당일 = 1.0 |
| 키워드 일치 | 30% | 제목 일치 +2점, 본문 일치 +1점 |
| 출처 신뢰도 | 20% | NIPA=1.0, Bizinfo=0.9, 웹=0.6 |

**규칙 보정**
- 공공기관 도메인 (`k-startup.go.kr`, `g2b.go.kr` 등) → +0.2 가점
- 목록/허브 URL (`/list`, `/search`, `/category` 등) → -0.5 강등

---

## 기술 스택

| 분류 | 기술 |
|---|---|
| Agent Framework | Google ADK 1.28.0 |
| LLM | GPT-4o-mini (LiteLLM 경유) |
| 벡터 검색 | FAISS 1.13.2 |
| 임베딩 | OpenAI `text-embedding-3-small` |
| 웹 검색 | Tavily Search API |
| 주가 데이터 | yfinance 0.2.41 |
| 데이터 검증 | Pydantic 2.12.5 |

---

## 프로젝트 구조

```
Industry-tailored-Agents/
├── apps/
│   └── root_app/
│       ├── agent.py        # 오케스트레이터 정의
│       └── prompts.py      # 라우팅 프롬프트
├── student/
│   ├── common/
│   │   ├── schemas.py      # Day1/2/3 데이터 스키마
│   │   ├── writer.py       # 마크다운 리포트 렌더러
│   │   └── fs_utils.py     # 파일 저장 유틸
│   ├── day1/
│   │   ├── agent.py        # Day1 에이전트 래퍼
│   │   └── impl/           # 웹 검색 / 주가 / 기업 개요 로직
│   ├── day2/
│   │   ├── agent.py        # Day2 에이전트 래퍼
│   │   └── impl/           # FAISS 인덱스 / RAG 로직
│   └── day3/
│       ├── agent.py        # Day3 에이전트 래퍼
│       └── impl/
│           ├── fetchers.py   # 멀티소스 수집
│           ├── normalize.py  # 스키마 정규화 & 중복 제거
│           └── rank.py       # 가중치 기반 스코어링
├── data/processed/         # 생성된 마크다운 리포트
├── indices/day2/           # FAISS 인덱스 파일
└── requirements.txt
```

---

## 설치 및 실행

### 1. 환경 설정

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env` 파일 생성:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
DAY2_INDEX_DIR=indices/day2
```

### 3. Day2 문서 인덱스 구축 (RAG 사용 시)

```bash
python -m student.day2.impl.ingest --paths data/your_documents/
```

### 4. 에이전트 실행

```bash
adk web apps
```

브라우저에서 `http://localhost:8000` 접속

---

## 출력 예시

### Day1 — 웹 리서치 리포트
```markdown
## 시세 스냅샷
- **AAPL**: 213.49 USD

## 기업 정보 요약
Apple Inc.는 소비자 전자기기, 소프트웨어 및 서비스를 설계, 제조 및 판매...

## 관련 링크 & 발췌
- [Apple 실적 발표](https://...) — Reuters (2026-01-30)
```

### Day3 — 공고 수집 결과
```markdown
| 출처 | 제목 | 기관 | 접수 마감 | 점수 |
|---|---|---|---|---|
| nipa | AI 바우처 지원사업 공고 | NIPA | 2026-04-30 | 0.821 |
| bizinfo | 헬스케어 스타트업 모집 | 기업마당 | 2026-04-15 | 0.763 |
```

---

## 핵심 설계 포인트

1. **멀티소스 수집 + 도메인 필터링** — 노이즈를 수집 단계에서 사전 차단
2. **공통 스키마 정규화** — 소스별 이질적 데이터를 표준 필드로 통일
3. **비즈니스 룰 기반 스코어링** — 마감 임박도 가중치로 실무 우선순위 반영
4. **RAG 신뢰도 게이트** — 유사도 점수 기반 fallback으로 답변 품질 보장
5. **오케스트레이터 라우팅** — 자연어 질의를 규칙 + LLM으로 자동 분류
