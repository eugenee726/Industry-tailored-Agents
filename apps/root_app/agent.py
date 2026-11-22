# -*- coding: utf-8 -*-
"""
루트 오케스트레이터
- 사용자 질의를 Day1(Web) / Day2(RAG) / Day3(Gov) 서브에이전트로 라우팅
"""
from __future__ import annotations
from typing import Optional

from google.adk.agents import Agent
from student.common.llm import adk_model

# 기존 서브 에이전트
from student.day1.agent import day1_web_agent
from student.day2.agent import day2_rag_agent
from student.day3.agent import day3_gov_agent


from .prompts import ORCHESTRATOR_DESC, ORCHESTRATOR_PROMPT

MODEL = adk_model()  # Claude Sonnet 5.5 (student/common/llm.py)

root_agent = Agent(
    name="Industry_Tailored_Orchestrator",
    model=MODEL,
    description=ORCHESTRATOR_DESC,
    instruction=ORCHESTRATOR_PROMPT,
    sub_agents=[
        day3_gov_agent,
        day2_rag_agent,
        day1_web_agent
    ],
)
