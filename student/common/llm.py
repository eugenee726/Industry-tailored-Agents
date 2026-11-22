# -*- coding: utf-8 -*-
"""
공통 LLM 설정 (Claude Sonnet 5.5)
- ADK 에이전트용 LiteLlm 모델과, 직접 호출용 Anthropic SDK 래퍼를 한곳에서 관리
- Sonnet 5.5는 temperature 등 샘플링 파라미터를 받지 않으므로 보내지 않는다.
  (재현성은 eval 쪽 캐시로 확보)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
ADK_MODEL_ID = f"anthropic/{CLAUDE_MODEL}"


def adk_model():
    """ADK Agent에 넘길 LiteLlm 모델 객체."""
    from google.adk.models.lite_llm import LiteLlm
    return LiteLlm(model=ADK_MODEL_ID)


@lru_cache(maxsize=1)
def _client():
    import anthropic
    return anthropic.Anthropic()


def claude_text(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 1024,
    effort: str = "low",
) -> str:
    """
    Claude에 단발성 요청을 보내고 텍스트만 반환.
    - effort 기본 low: 분류/요약/채점처럼 짧은 작업의 비용을 줄이기 위함
    - 안전 분류기가 거절하면 서버 측 fallback 모델이 이어받도록 설정
    - 최종적으로 거절되면 RuntimeError
    """
    kwargs = dict(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    resp = _client().beta.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Claude refused: {getattr(resp, 'stop_details', None)}")
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def claude_json(
    prompt: str,
    schema: dict,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    effort: str = "low",
) -> dict:
    """구조화 출력(JSON schema)으로 받아 dict 반환. 채점/데이터 생성용."""
    import json

    kwargs = dict(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    resp = _client().beta.messages.create(**kwargs)
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Claude refused: {getattr(resp, 'stop_details', None)}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Claude output truncated (max_tokens)")
    return json.loads("".join(b.text for b in resp.content if b.type == "text"))
