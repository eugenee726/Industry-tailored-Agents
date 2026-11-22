# -*- coding: utf-8 -*-
"""
raw → GovNotice 표준 스키마 정규화
- fetchers.py에서 온 Day1형 raw 결과를 GovNotice 필드로 매핑
- URL 중복 제거
"""
import re
from typing import List, Dict, Optional
from datetime import date, datetime

DATE_FMTS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y-%m-%dT%H:%M:%S%z")


def _as_date_iso(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    for fmt in DATE_FMTS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            continue
    # 숫자 8자리(YYYYMMDD) 대응
    if s.isdigit() and len(s) == 8:
        try:
            return datetime.strptime(s, "%Y%m%d").date().isoformat()
        except Exception:
            pass
    return ""


# ── 마감일 추출 ────────────────────────────────────────────────────────────────
# 이전에는 close_date를 항상 ""로 두어, 랭킹 가중치 50%인 '마감 임박도'가 한 번도 작동하지 않았다.
# 제목/스니펫의 날짜 표현에서 마감일을 규칙 기반으로 추출한다.
_D = r"(\d{4})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})\s*일?\.?"      # 2026.04.30 / 2026년 4월 30일
_MD = r"(\d{1,2})\s*[./월]\s*(\d{1,2})\s*일?"                                      # 4월 30일 / 4.30
_DOW = r"(?:\s*\([월화수목금토일]\))?"
_RE_RANGE = re.compile(_D + _DOW + r"[^~\n]{0,15}?[~∼\-–]\s*(?:" + _D + r"|" + _MD + r")")
_RE_DEADLINE_FULL = re.compile(r"(?:마감|접수\s*마감|신청\s*마감|마감일|기한)\s*[:：]?\s*" + _D)
_RE_UNTIL_FULL = re.compile(_D + _DOW + r"\s*(?:\d{1,2}:\d{2}\s*)?까지")
_RE_DEADLINE_MD = re.compile(r"(?:마감|마감일|기한)\s*[:：]?\s*" + _MD)
_RE_OPEN_END_MD = re.compile(r"[~∼]\s*" + _MD + r"(?!\d)")
_RE_UNTIL_MD = re.compile(_MD + _DOW + r"\s*(?:\d{1,2}:\d{2}\s*)?까지")


def _mk(y, m, d) -> Optional[str]:
    try:
        return date(int(y), int(m), int(d)).isoformat()
    except (TypeError, ValueError):
        return None


def extract_close_date(text: str, ref_year: Optional[int] = None) -> str:
    """
    텍스트에서 접수 마감일(YYYY-MM-DD)을 추출. 못 찾으면 "".
    우선순위: ① '마감/기한' 뒤 날짜 ② 기간 표현의 끝 날짜 ③ '~까지' 앞 날짜 ④ 연도 없는 표현
    연도가 없는 '4월 30일까지'는 ref_year(공고일 연도, 없으면 올해)로 보정.
    """
    if not text:
        return ""
    ref_year = ref_year or date.today().year
    # ① '마감/기한'이 명시된 날짜가 최우선 (예: "사업기간 ... / 접수마감: 2026.04.25")
    m = _RE_DEADLINE_FULL.search(text)
    if m:
        return _mk(*m.groups()[:3]) or ""
    # ② 기간 표현의 끝 날짜
    m = _RE_RANGE.search(text)
    if m:
        g = m.groups()
        if g[3]:                             # 끝 날짜에 연도 있음
            return _mk(g[3], g[4], g[5]) or ""
        return _mk(g[0], g[6], g[7]) or ""   # 끝 날짜는 월/일만 → 시작 연도 사용
    # ③ '~까지' 앞 날짜
    m = _RE_UNTIL_FULL.search(text)
    if m:
        return _mk(*m.groups()[:3]) or ""
    # ④ 연도 없는 표현: '마감 5월 15일', '4월 30일까지', '(~4.20)'
    for rx in (_RE_DEADLINE_MD, _RE_UNTIL_MD, _RE_OPEN_END_MD):
        m = rx.search(text)
        if m:
            return _mk(ref_year, *m.groups()[:2]) or ""
    return ""


def normalize_all(raw_items: List[Dict]) -> List[Dict]:
    norm: List[Dict] = []
    for r in raw_items or []:
        # Day1 웹결과 스키마: title/url/source/snippet/date
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        source = (r.get("source") or "").strip().lower()
        snippet = (r.get("snippet") or "").strip()
        date_guess = _as_date_iso(r.get("date") or "")

        norm.append({
            "title": title,
            "url": url,
            "source": "nipa" if "nipa" in source else ("bizinfo" if "bizinfo" in source else "web"),
            "agency": "",
            "announce_date": date_guess,   # 알 수 없으면 빈 값
            "close_date": extract_close_date(f"{title} {snippet}",
                                             int(date_guess[:4]) if date_guess else None),
            "budget": "",
            "snippet": snippet,
            "attachments": [],
            "content_type": "notice",
            "score": 0.0,
        })

    # URL 기준 중복 제거
    seen = set()
    deduped = []
    for n in norm:
        u = n["url"]
        if not u or u in seen:
            continue
        seen.add(u)
        deduped.append(n)
    return deduped
