# -*- coding: utf-8 -*-
"""
Day3 스모크 테스트 (루트 .env 로드 + sys.path 보정 + 전체 파이프라인 검증)
- 이 파일만 수정/실행합니다. 배포된 모듈은 건드리지 않습니다.
"""
# --- 0) 프로젝트 루트 탐색 + sys.path 보정 + .env 로드 ---
import os, sys, json
from pathlib import Path

def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists() or (p / "apps").exists():
            return p
    return start

ROOT = _find_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENV_PATH = ROOT / ".env"

def _manual_load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH, override=False)
except Exception:
    _manual_load_env(ENV_PATH)

# ------------------------------------------------------------------------------
print("=" * 80)
print("Day3 Smoke Test - 정부사업 공고 에이전트 검증")
print("=" * 80)

# --- 1) 환경변수 체크 ---
def _check_keys() -> bool:
    ok = True
    tavily_key = os.getenv("TAVILY_API_KEY")
    public_key = os.getenv("PUBLIC_DATA_API_KEY")
    
    print("\n[1단계] 환경변수 확인")
    print("-" * 80)
    
    if not tavily_key:
        print("❌ TAVILY_API_KEY가 설정되지 않았습니다.")
        ok = False
    else:
        print(f"✅ TAVILY_API_KEY: {tavily_key[:10]}...{tavily_key[-5:]}")
    
    if not public_key:
        print("⚠️  PUBLIC_DATA_API_KEY가 설정되지 않았습니다. (선택사항)")
    else:
        print(f"✅ PUBLIC_DATA_API_KEY: {public_key[:10]}...{public_key[-5:]}")
    
    return ok

if not _check_keys():
    print("\n" + "=" * 80)
    print("❌ 필수 환경변수가 없습니다. 루트 .env를 확인하세요.")
    print("=" * 80)
    sys.exit(2)

# --- 2) 모듈 Import 테스트 ---
print("\n[2단계] 모듈 Import 테스트")
print("-" * 80)

try:
    from student.day3.impl import fetchers
    print("✅ fetchers 모듈")
except ImportError as e:
    print(f"❌ fetchers 모듈 로딩 실패: {e}")
    sys.exit(1)

try:
    from student.day3.impl.normalize import normalize_all
    print("✅ normalize 모듈")
except ImportError as e:
    print(f"❌ normalize 모듈 로딩 실패: {e}")
    sys.exit(1)

try:
    from student.day3.impl.rank import rank_items
    print("✅ rank 모듈")
except ImportError as e:
    print(f"❌ rank 모듈 로딩 실패: {e}")
    sys.exit(1)

try:
    from student.day3.impl.agent import Day3Agent
    print("✅ Day3Agent")
except ImportError as e:
    print(f"❌ Day3Agent 로딩 실패: {e}")
    sys.exit(1)

try:
    from student.common.schemas import Day3Plan
    print("✅ Day3Plan")
except ImportError as e:
    print(f"❌ Day3Plan 로딩 실패: {e}")
    sys.exit(1)

# --- 3) Fetchers 테스트 ---
print("\n[3단계] Fetchers 기능 테스트")
print("-" * 80)

test_query = "AI 바우처 지원사업"
print(f"테스트 쿼리: {test_query}\n")

# 3-1) NIPA 검색
try:
    nipa_results = fetchers.fetch_nipa(test_query, topk=2)
    print(f"✅ NIPA 검색: {len(nipa_results)}건")
    if nipa_results:
        print(f"   샘플: {nipa_results[0].get('title', 'N/A')[:60]}")
except Exception as e:
    print(f"⚠️  NIPA 검색 실패: {e}")

# 3-2) Bizinfo 검색
try:
    biz_results = fetchers.fetch_bizinfo(test_query, topk=2)
    print(f"✅ Bizinfo 검색: {len(biz_results)}건")
    if biz_results:
        print(f"   샘플: {biz_results[0].get('title', 'N/A')[:60]}")
except Exception as e:
    print(f"⚠️  Bizinfo 검색 실패: {e}")

# 3-3) 웹 검색
try:
    web_results = fetchers.fetch_web(test_query, topk=2)
    print(f"✅ 웹 검색: {len(web_results)}건")
    if web_results:
        print(f"   샘플: {web_results[0].get('title', 'N/A')[:60]}")
except Exception as e:
    print(f"⚠️  웹 검색 실패: {e}")

# 3-4) 전체 수집
try:
    all_results = fetchers.fetch_all(test_query)
    print(f"✅ 전체 수집: {len(all_results)}건")
except Exception as e:
    print(f"❌ 전체 수집 실패: {e}")
    sys.exit(1)

# --- 4) Normalize 테스트 ---
print("\n[4단계] Normalize 기능 테스트")
print("-" * 80)

try:
    normalized = normalize_all(all_results)
    print(f"✅ 정규화 완료: {len(normalized)}건 (중복 제거 후)")
    
    if normalized:
        sample = normalized[0]
        print(f"   샘플 구조:")
        print(f"   - title: {sample.get('title', 'N/A')[:50]}")
        print(f"   - url: {sample.get('url', 'N/A')[:50]}")
        print(f"   - source: {sample.get('source', 'N/A')}")
        print(f"   - snippet: {sample.get('snippet', 'N/A')[:50]}")
except Exception as e:
    print(f"❌ 정규화 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 5) Rank 테스트 ---
print("\n[5단계] Rank 기능 테스트")
print("-" * 80)

try:
    ranked = rank_items(normalized, test_query)
    print(f"✅ 랭킹 완료: {len(ranked)}건")
    
    if ranked:
        print(f"\n   상위 3개:")
        for i, item in enumerate(ranked[:3], 1):
            score = item.get('score', 0.0)
            title = item.get('title', 'N/A')[:60]
            print(f"   [{i}] (점수: {score:.2f}) {title}")
except Exception as e:
    print(f"❌ 랭킹 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 6) Day3Agent 전체 파이프라인 테스트 ---
print("\n[6단계] Day3Agent 전체 파이프라인 테스트")
print("-" * 80)

try:
    # Plan 생성
    plan = Day3Plan(
        nipa_topk=3,
        bizinfo_topk=2,
        web_topk=2,
        use_web_fallback=True
    )
    print(f"✅ Day3Plan 생성: NIPA={plan.nipa_topk}, Bizinfo={plan.bizinfo_topk}, Web={plan.web_topk}")
    
    # Agent 생성
    agent = Day3Agent()
    print("✅ Day3Agent 인스턴스 생성")
    
    # Handle 실행
    payload = agent.handle(test_query, plan)
    print(f"✅ Agent.handle() 실행 완료")
    
    # Payload 검증
    assert payload.get("type") == "gov_notices", "❌ Payload type 오류"
    assert payload.get("query") == test_query, "❌ Query 불일치"
    assert "items" in payload, "❌ items 필드 없음"
    
    items = payload.get("items", [])
    print(f"✅ Payload 검증 완료: {len(items)}건의 공고")
    
    # 결과 샘플 출력
    if items:
        print(f"\n   결과 샘플 (상위 3개):")
        for i, item in enumerate(items[:3], 1):
            print(f"   [{i}] {item.get('title', 'N/A')[:70]}")
            print(f"       출처: {item.get('source', 'N/A')}")
            print(f"       URL: {item.get('url', 'N/A')[:60]}")
            print()
    else:
        print("   ⚠️  결과가 없습니다. (검색어 또는 API 상태 확인 필요)")
    
    # JSON 출력 (선택)
    print("\n   Payload JSON (일부):")
    payload_sample = {
        "type": payload.get("type"),
        "query": payload.get("query"),
        "total": len(payload.get("items", [])),
        "sources": payload.get("sources", {}),
    }
    print(f"   {json.dumps(payload_sample, ensure_ascii=False, indent=2)}")

except AssertionError as ae:
    print(f"❌ Payload 검증 실패: {ae}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Agent 실행 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 7) 파일 생성 테스트 (선택) ---
print("\n[7단계] 마크다운 렌더링 테스트 (선택)")
print("-" * 80)

try:
    from student.common.writer import render_day3
    
    body_md = render_day3(test_query, payload)
    print(f"✅ 마크다운 렌더링 완료: {len(body_md)}자")
    print(f"\n   미리보기 (처음 300자):")
    print("   " + "-" * 76)
    print("   " + body_md[:300].replace("\n", "\n   "))
    print("   " + "-" * 76)
except Exception as e:
    print(f"⚠️  마크다운 렌더링 실패: {e}")

# --- 최종 결과 ---
print("\n" + "=" * 80)
print("✅ Day3 Smoke Test 통과!")
print("=" * 80)
print("\n다음 단계:")
print("  1. Day3 에이전트 실행: python -m student.day3.agent")
print("  2. 실제 질의 테스트: 'AI 지원사업', '헬스케어 공고' 등")
print("  3. Day4 통합 준비")
print("=" * 80)