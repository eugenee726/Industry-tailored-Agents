# -*- coding: utf-8 -*-
"""
Day3: 정부사업 공고 에이전트
- 역할: 사용자 질의를 받아 Day3 본체(impl/agent.py)의 Day3Agent.handle을 호출
- 결과를 writer로 표/요약 마크다운으로 렌더 → 파일 저장(envelope 포함) → LlmResponse 반환
"""

from __future__ import annotations
from typing import Dict, Any, Optional

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

# Day3 본체
from student.day3.impl.agent import Day3Agent
# 공용 렌더/저장/스키마
from student.common.fs_utils import save_markdown
from student.common.writer import render_day3, render_enveloped
from student.common.schemas import Day3Plan



MODEL = LiteLlm(model="openai/gpt-4o-mini")


def _handle(query: str) -> Dict[str, Any]:

    plan = Day3Plan(
        nipa_topk=3,
        bizinfo_topk=2,
        web_topk=2,
        use_web_fallback=True,
        # (옵션) 랭킹/스코어링 파라미터가 있다면 여기서 지정 가능
        # deadline_weight=0.5, keyword_weight=0.3, source_weight=0.2,
    )
    agent = Day3Agent()
    payload = agent.handle(query, plan)
    return payload



def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    **kwargs,
) -> Optional[LlmResponse]:
    # 정답 구현:
    try:
        last = llm_request.contents[-1]
        if last.role == "user":
            query = last.parts[0].text
            payload = _handle(query)

            body_md = render_day3(query, payload)
            saved = save_markdown(query=query, route="day3", markdown=body_md)
            md = render_enveloped(kind="day3", query=query, payload=payload, saved_path=saved)

            return LlmResponse(
                content=types.Content(parts=[types.Part(text=md)], role="model")
            )
    except Exception as e:
        return LlmResponse(
            content=types.Content(parts=[types.Part(text=f"Day3 에러: {e}")], role="model")
        )
    return None



day3_gov_agent = Agent(
    name="Day3GovAgent",                        # <- 필요 시 수정
    model=MODEL,                                # <- TODO[DAY3-A-01]에서 설정
    description="정부사업 공고/바우처 정보 수집 및 표 제공",   # <- 필요 시 수정
    instruction="질의를 기반으로 정부/공공 포털에서 관련 공고를 수집하고 표로 요약해라.",
    tools=[],
    before_model_callback=before_model_callback,
)
