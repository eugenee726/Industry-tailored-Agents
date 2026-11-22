# -*- coding: utf-8 -*-
"""
평가 공통 유틸
- .env 로드, 경로 상수
- JSONL 캐시: 같은 입력이면 API를 다시 호출하지 않음 (재현성 + 비용 0 재실행)
- 데이터셋 로드 / 결과 CSV 저장
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

EVAL_DIR = ROOT / "eval"
DATA_DIR = EVAL_DIR / "datasets"
RESULT_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / "cache"
for d in (DATA_DIR, RESULT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip() and not l.lstrip().startswith("//")]


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class JsonlCache:
    """key(dict) -> value(any JSON) 캐시. 파일 하나에 append-only로 쌓는다."""

    def __init__(self, name: str):
        self.path = CACHE_DIR / f"{name}.jsonl"
        self.data: Dict[str, Any] = {}
        self.hits = 0
        self.misses = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["k"]] = rec["v"]

    @staticmethod
    def _key(key: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def get_or_call(self, key: Dict[str, Any], fn: Callable[[], Any]) -> Any:
        k = self._key(key)
        if k in self.data:
            self.hits += 1
            return self.data[k]
        self.misses += 1
        v = fn()
        self.data[k] = v
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "key": key, "v": v}, ensure_ascii=False) + "\n")
        return v

    def stats(self) -> str:
        return f"[cache:{self.path.name}] hit={self.hits} api_call={self.misses}"


def save_csv(rows: List[Dict[str, Any]], name: str):
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_DIR / name, index=False, encoding="utf-8-sig")
    return df
