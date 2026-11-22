# -*- coding: utf-8 -*-
"""
ADK 하위 에이전트 공통 유틸

user_query: 하위 에이전트가 처리할 '사용자 질의'를 꺼낸다.
- 오케스트레이터가 transfer하면 ADK는 대화 끝에 "For context: [오케스트레이터] called tool ..." 형태의
  user-role 메시지를 덧붙인다. 그래서 llm_request.contents[-1]을 질의로 쓰면 "For context:"를 검색하게 된다.
- 이번 호출(invocation)을 시작한 사용자 메시지(callback_context.user_content)를 우선 사용한다.
"""
from __future__ import annotations

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest

_ADK_CONTEXT_PREFIX = "For context:"


def _text_of(content) -> str:
    if not content or not getattr(content, "parts", None):
        return ""
    return " ".join(p.text for p in content.parts if getattr(p, "text", None)).strip()


def user_query(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[str]:
    q = _text_of(callback_context.user_content)
    if q:
        return q
    # 폴백: ADK가 덧붙인 컨텍스트 메시지를 건너뛰고 가장 최근의 실제 사용자 메시지
    for content in reversed(llm_request.contents or []):
        if content.role != "user":
            continue
        t = _text_of(content)
        if t and not t.startswith(_ADK_CONTEXT_PREFIX):
            return t
    return None
