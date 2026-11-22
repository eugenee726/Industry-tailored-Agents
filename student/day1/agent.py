# -*- coding: utf-8 -*-
"""
Day1: 웹+주가+기업개요 에이전트
- 역할: 사용자 질의를 받아 Day1 본체 호출 → 결과 렌더 → 파일 저장(envelope) → 응답
- 본 파일은 "UI용 래퍼"로, 실질적인 수집/요약 로직은 impl/agent.py 등에 있음.

"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import os
import re

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from student.common.llm import adk_model
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from student.common.adk_utils import user_query

from student.common.schemas import Day1Plan
from student.common.writer import render_day1, render_enveloped
from student.common.fs_utils import save_markdown
from student.day1.impl.agent import Day1Agent
from student.day1.impl.web_search import looks_like_ticker

MODEL = adk_model()  # Claude Sonnet 5.5 (student/common/llm.py)


# 대문자 약어지만 종목 티커가 아닌 단어들 (오인식 방지)
_NON_TICKER_WORDS = {
    "AI", "CEO", "CTO", "CFO", "IT", "ETF", "ESG", "IPO", "FDA", "USA", "US", "UK", "EU", "KR",
    "SW", "HW", "API", "LLM", "GPT", "RAG", "PDF", "RFP", "NIPA", "KCI", "CES", "SAAS", "MOU",
    "IOT", "VR", "AR", "XR", "DX", "AX", "GDP", "CPI", "PER", "PBR", "ROE", "EPS", "KOSPI",
    "MLMD", "SAMD", "UI", "UX", "QA", "PM", "HR", "B2B", "B2C", "OK", "PPS", "KEIT", "NTIS",
}
_STOCK_INTENT = re.compile(r"주가|시세|주식|종가|시총|시가총액|티커|stock|price|quote", re.I)


def _extract_tickers_from_query(query: str) -> List[str]:
    """
    사용자 질의에서 티커 후보를 추출한다.
      - "AAPL 주가 알려줘"      → ["AAPL"]
      - "삼성전자 005930 분석"  → ["005930"]
      - "NVDA/TSLA 비교"       → ["NVDA", "TSLA"]
      - "ai 의학에 대해 알려줘" → []   (이전 구현은 query.upper() 때문에 "AI"를 티커로 오인식)

    규칙:
      1) 원문에서 '대문자로 쓴' 영문 1~5자 토큰만 후보 (소문자 단어를 대문자화하지 않음)
      2) 한글이 바로 붙은 경우도 허용 ("AAPL주가")
      3) 흔한 약어(AI, CEO, FDA, NIPA...)는 제외
      4) 1글자 토큰(F, T 등)은 '주가/시세' 같은 주식 의도 단어가 있을 때만 인정 ("R&D" 오인식 방지)
      5) 6자리 숫자는 국내 종목코드
    """
    has_intent = bool(_STOCK_INTENT.search(query))
    alpha_hits = re.findall(r"(?<![A-Za-z0-9&])[A-Z]{1,5}(?![A-Za-z0-9&])", query)
    alpha_hits = [
        t for t in alpha_hits
        if t not in _NON_TICKER_WORDS and (len(t) >= 2 or has_intent)
    ]
    digit_hits = re.findall(r"(?<!\d)\d{6}(?!\d)", query)

    deduped: List[str] = []
    for sym in alpha_hits + digit_hits:
        if sym not in deduped:
            deduped.append(sym)
    return deduped


def _normalize_kr_tickers(tickers: List[str]) -> List[str]:
    """
    한국식 6자리 종목코드에 '.KS'를 붙여 yfinance 호환 심볼로 보정합니다.
    예:
      ['005930', 'AAPL'] → ['005930.KS', 'AAPL']
    구현 포인트:
      1) 각 원소가 6자리 숫자면 뒤에 '.KS'를 붙임
      2) 이미 확장자가 붙은 경우(예: '.KS')는 그대로 둠
    """

    normalized: List[str] = []
    for sym in tickers:
        if re.fullmatch(r"\d{6}", sym):
            normalized.append(f"{sym}.KS")
        else:
            normalized.append(sym)
    return normalized


def _handle(query: str) -> Dict[str, Any]:
    """
    Day1 전체 흐름(오케스트레이션):
      1) 키 준비: os.getenv("TAVILY_API_KEY", "")
      2) 티커 추출 → 한국형 보정
      3) Day1Plan 구성
         - do_web=True (웹 검색은 기본 수행)
         - do_stocks=True/False (티커가 존재하면 True)
         - web_keywords: [query] (필요시 키워드 가공 가능)
         - tickers: 보정된 티커 리스트
      4) Day1Agent(tavily_api_key=...) 인스턴스 생성
      5) agent.handle(query, plan) 호출 → payload(dict) 수신
    반환:
      merge된 표준 스키마 dict (impl/merge.py 참고)
    """

    import os
    from student.day1.impl.web_search import looks_like_ticker

    # 1) API 키
    api_key = os.getenv("TAVILY_API_KEY", "")

    # 2) 티커 추출 → 한국형 보정 → 실제 티커처럼 보이는 것만 남김
    raw = _extract_tickers_from_query(query)
    normalized = _normalize_kr_tickers(raw)
    tickers = [t for t in normalized if looks_like_ticker(t)]

    # 3) 계획 구성
    plan = Day1Plan(
        do_web=True,
        do_stocks=bool(tickers),
        web_keywords=[query],
        tickers=tickers,
        output_style="report",
    )

    # 4) 에이전트 생성
    agent = Day1Agent(
        tavily_api_key=api_key,
        web_topk=6,
        request_timeout=20,
    )

    # 5) 실행
    return agent.handle(query, plan)



def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    **kwargs,
) -> Optional[LlmResponse]:
    """
    UI 엔트리포인트:
      1) llm_request.contents[-1]에서 사용자 메시지 텍스트(query) 추출
      2) _handle(query) 호출 → payload 획득
      3) 본문 마크다운 렌더: render_day1(query, payload)
      4) 저장: save_markdown(query, route='day1', markdown=본문MD) → 경로
      5) envelope: render_enveloped('day1', query, payload, saved_path)
      6) LlmResponse로 반환
      7) 예외시 간단한 오류 텍스트 반환
    """
 
    try:
        query = user_query(callback_context, llm_request)
        if query:
            payload = _handle(query)

            body_md = render_day1(query, payload)
            saved_path = save_markdown(query=query, route="day1", markdown=body_md)
            enveloped_md = render_enveloped(
                kind="day1",
                query=query,
                payload=payload,
                saved_path=saved_path,
            )

            return LlmResponse(
                content=types.Content(
                    parts=[types.Part(text=enveloped_md)],
                    role="model",
                )
            )
    except Exception as e:
 
        return LlmResponse(
            content=types.Content(
                parts=[types.Part(text=f"Day1 에러: {e}")],
                role="model",
            )
        )
    return None



day1_web_agent = Agent(
    name="Day1WebAgent",
    model=MODEL,
    description="웹 검색 + (필요시) 주가 + 기업 개요 요약 제공",
    instruction=(
        "사용자 질의에 대해 웹 검색 결과와 (티커가 있으면) 주가 스냅샷, "
        "기업 개요 요약을 표준 마크다운으로 제공하라. 필요한 경우 출처 링크를 포함하라."
    ),
    tools=[],
    before_model_callback=before_model_callback,
    # 답변 후 다음 질문은 다시 오케스트레이터가 라우팅하도록 (ADK 기본값은 넘겨받은 에이전트가 계속 응답)
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)
