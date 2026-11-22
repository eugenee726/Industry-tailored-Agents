# -*- coding: utf-8 -*-
"""
Day2: RAG 도구 에이전트
- 역할: Day2 RAG 본체 호출 → 결과 렌더 → 저장(envelope) → 응답

"""

from __future__ import annotations
from typing import Dict, Any
import os

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from student.common.llm import adk_model
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from student.day2.impl.rag import Day2Agent
from student.common.writer import render_day2, render_enveloped
from student.common.schemas import Day2Plan
from student.common.fs_utils import save_markdown


MODEL = adk_model()  # Claude Sonnet 5.5 (student/common/llm.py)
_AGENT = Day2Agent()  # 인덱스/BM25를 한 번만 로드해 재사용


def _handle(query: str) -> Dict[str, Any]:
    """기본 설정(Day2Plan 기본값 = 평가로 정한 값)으로 RAG 실행"""
    plan = Day2Plan(index_dir=os.getenv("DAY2_INDEX_DIR", "indices/day2"))
    return _AGENT.handle(query, plan)


def before_model_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    **kwargs,
) -> LlmResponse | None:
    """
    1) 사용자 메시지에서 query 텍스트 추출
    2) payload = _handle(query)
    3) body_md = render_day2(query, payload)
    4) saved = save_markdown(query, 'day2', body_md)
    5) md = render_enveloped('day2', query, payload, saved)
    6) LlmResponse로 반환 (예외 발생 시 간단 메시지)
    """

    try:
        last = llm_request.contents[-1]
        if last.role == "user":
            query = last.parts[0].text
            payload = _handle(query)

            body_md = render_day2(query, payload)
            saved = save_markdown(query=query, route="day2", markdown=body_md)
            md = render_enveloped(kind="day2", query=query, payload=payload, saved_path=saved)

            return LlmResponse(
                content=types.Content(
                    parts=[types.Part(text=md)],
                    role="model",
                )
            )
    except Exception as e:
        return LlmResponse(
            content=types.Content(
                parts=[types.Part(text=f"Day2 에러: {e}")],
                role="model",
            )
        )
    return None


day2_rag_agent = Agent(
    name="Day2RagAgent",
    model=MODEL,
    description="로컬 인덱스를 활용한 RAG 요약/근거 제공",
    instruction="사용자 질의와 관련된 문서를 인덱스에서 찾아 요약하고 근거를 함께 제시하라.",
    tools=[],
    before_model_callback=before_model_callback,
)
