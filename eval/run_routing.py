# -*- coding: utf-8 -*-
"""
라우팅 평가: 오케스트레이터가 질의를 올바른 서브에이전트로 보내는가?

- 실제 root_agent(ADK + Claude)를 그대로 실행한다.
- 서브에이전트의 before_model_callback은 고정 텍스트를 반환하는 스텁으로 교체
  → 웹검색/임베딩 비용 0, 오케스트레이터 LLM 호출만 발생.
- 첫 번째 transfer_to_agent 대상(없으면 "NONE")을 예측값으로 기록.
- 같은 질의를 N회 반복 실행해 일관성(일치율)도 측정.

사용법:
  python -m eval.run_routing --prompt v1 --repeats 3
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter

from eval.common import DATA_DIR, JsonlCache, RESULT_DIR, load_jsonl, save_csv
from eval.metrics import accuracy, classification_report, confusion

LABELS = ["Day1WebAgent", "Day2RagAgent", "Day3GovAgent"]


def _build_agent(prompt_version: str):
    from google.genai import types
    from google.adk.models.llm_response import LlmResponse
    from apps.root_app import prompts
    from apps.root_app.agent import root_agent

    v = prompt_version.upper()
    root_agent.instruction = getattr(prompts, f"ORCHESTRATOR_PROMPT_{v}")
    root_agent.description = getattr(prompts, f"ORCHESTRATOR_DESC_{v}")

    def stub(name):
        def cb(callback_context, llm_request, **kw):
            return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=f"[stub:{name}]")]))
        return cb

    for a in root_agent.sub_agents:
        a.before_model_callback = stub(a.name)
    return root_agent


async def _route_once(agent, query: str) -> str:
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=agent, app_name="routing_eval")
    sess = await runner.session_service.create_session(app_name="routing_eval", user_id="eval")
    msg = types.Content(role="user", parts=[types.Part(text=query)])
    async for ev in runner.run_async(user_id="eval", session_id=sess.id, new_message=msg):
        if ev.actions and ev.actions.transfer_to_agent:
            return ev.actions.transfer_to_agent
    return "NONE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--data", default="routing", help="routing(개발셋) | routing_holdout(검증셋)")
    args = ap.parse_args()
    tag = args.prompt if args.data == "routing" else f"{args.prompt}_{args.data.replace('routing_', '')}"

    data = load_jsonl(DATA_DIR / f"{args.data}.jsonl")
    agent = _build_agent(args.prompt)
    cache = JsonlCache("routing")
    print(f"[routing] prompt={args.prompt} queries={len(data)} repeats={args.repeats} "
          f"(최대 API 호출 {len(data) * args.repeats}회, 캐시 적중분 제외)")

    # 이벤트 루프 하나로 전체 실행 (호출마다 asyncio.run을 쓰면 Windows에서 소켓이 고갈됨)
    async def collect():
        out = {}
        for d in data:
            for r in range(args.repeats):
                key = {"prompt": args.prompt, "query": d["query"], "rep": r}
                k = JsonlCache._key(key)
                if k in cache.data:
                    cache.hits += 1
                    out[(d["id"], r)] = cache.data[k]
                else:
                    pred = await _route_once(agent, d["query"])
                    out[(d["id"], r)] = cache.get_or_call(key, lambda: pred)
        return out

    routed = asyncio.run(collect())

    rows = []
    for i, d in enumerate(data, 1):
        preds = [routed[(d["id"], r)] for r in range(args.repeats)]
        majority, cnt = Counter(preds).most_common(1)[0]
        rows.append({
            "id": d["id"], "query": d["query"], "expected": d["expected"],
            "difficulty": d.get("difficulty", ""), "pred": preds[0], "majority": majority,
            "consistent": cnt == len(preds), "all_preds": "|".join(preds),
            "correct": preds[0] == d["expected"],
        })
        print(f"  {i:02d}/{len(data)} {d['id']} → {preds} {'OK' if preds[0] == d['expected'] else 'MISS'}")
    print(cache.stats())

    df = save_csv(rows, f"routing_{tag}.csv")
    y_true, y_pred = df["expected"].tolist(), df["pred"].tolist()
    acc = accuracy(y_true, y_pred)
    rep = classification_report(y_true, y_pred, LABELS)
    cm = confusion(y_true, y_pred, LABELS)
    by_diff = df.groupby("difficulty")["correct"].agg(["mean", "count"]).rename(columns={"mean": "accuracy"})
    consistency = df["consistent"].mean()

    rep.to_csv(RESULT_DIR / f"routing_{tag}_report.csv", index=False, encoding="utf-8-sig")
    cm.to_csv(RESULT_DIR / f"routing_{tag}_confusion.csv", encoding="utf-8-sig")
    by_diff.to_csv(RESULT_DIR / f"routing_{tag}_by_difficulty.csv", encoding="utf-8-sig")

    print(f"\n== Routing {args.prompt} ==")
    print(f"Accuracy: {acc:.3f}  |  {args.repeats}회 반복 일관성: {consistency:.3f}")
    print(rep.round(3).to_string(index=False))
    print("\nConfusion matrix\n", cm.to_string())
    print("\nBy difficulty\n", by_diff.round(3).to_string())
    miss = df[~df["correct"]]
    if len(miss):
        print("\nMisrouted:")
        for _, m in miss.iterrows():
            print(f"  [{m['id']}] {m['query']}  expected={m['expected']} pred={m['pred']}")


if __name__ == "__main__":
    main()
