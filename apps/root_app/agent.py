# -*- coding: utf-8 -*-
"""
루트 오케스트레이터
- 사용자 질의를 Day1(Web) / Day2(RAG) / Day3(Gov) 서브에이전트로 라우팅
"""
from __future__ import annotations
import re
from typing import Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from student.common.llm import adk_model

# 기존 서브 에이전트
from student.day1.agent import day1_web_agent
from student.day2.agent import day2_rag_agent
from student.day3.agent import day3_gov_agent


from .prompts import ORCHESTRATOR_DESC, ORCHESTRATOR_PROMPT

MODEL = adk_model()  # Claude Sonnet 5.5 (student/common/llm.py)

# 인사·감사 같은 짧은 잡담은 라우팅 대상이 아님.
# v2 프롬프트는 "직접 답변 금지"라 LLM이 인사를 웹 검색으로 보내는 경우가 있어, 규칙으로 먼저 처리한다.
# (라우팅 평가셋에는 인사 질의가 없으므로 평가 수치에는 영향 없음)
_SMALL_TALK = re.compile(r"^\s*(안녕(하세요|하십니까)?|하이|헬로|ㅎㅇ|hi|hello|hey|반가워(요)?|반갑습니다|고마워(요)?|감사(합니다|해요)?|ㄱㅅ|thanks|thank you|도움말|help)(\s+there)?[\s!~.?ㅎㅋ^]*$", re.I)
_GUIDE = (
    "안녕하세요! 질문을 주시면 알맞은 담당 에이전트로 연결해 드립니다.\n\n"
    "- **정부 지원사업 · 공고 · 입찰**: 예) `NIPA AI 바우처 지원사업 공고 찾아줘`\n"
    "- **보유 문서 (AI 의료기기 규제 · 법적 책임)**: 예) `AI 의료기기 오작동 시 누가 책임져?`\n"
    "- **기업 뉴스 · 주가 · 시장 동향**: 예) `엔비디아 주가랑 최근 뉴스 알려줘`"
)


def _small_talk_guard(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    content = callback_context.user_content
    text = " ".join(p.text for p in (content.parts if content else []) if getattr(p, "text", None)).strip()
    if text and len(text) <= 15 and _SMALL_TALK.match(text):
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=_GUIDE)]))
    return None


root_agent = Agent(
    name="Industry_Tailored_Orchestrator",
    model=MODEL,
    description=ORCHESTRATOR_DESC,
    instruction=ORCHESTRATOR_PROMPT,
    before_model_callback=_small_talk_guard,
    sub_agents=[
        day3_gov_agent,
        day2_rag_agent,
        day1_web_agent
    ],
)
