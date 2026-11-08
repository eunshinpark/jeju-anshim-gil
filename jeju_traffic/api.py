"""제주데이터허브 API 수집: 교통사고 다발지역 정보."""
from __future__ import annotations

import re
import time
from datetime import datetime

import pandas as pd
import requests

from .config import PAGE_SIZE


class ApiError(RuntimeError):
    pass


def _get(session, url, params, retries=4):
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 403:
                raise ApiError("403 Forbidden - API 사용 승인 또는 일일 호출 한도를 확인하세요.")
            r.raise_for_status()
            return r.json()
        except ApiError:
            raise
        except (requests.RequestException, ValueError):
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def fetch_all(url: str, delay: float = 0.15) -> list[dict]:
    rows, n = [], 1
    with requests.Session() as s:
        while True:
            d = _get(s, url, {"limit": PAGE_SIZE, "number": n})
            rows += d.get("data", [])
            if not d.get("hasMore"):
                break
            n += 1
            time.sleep(delay)
        if len(rows) != d.get("totCnt", len(rows)):
            raise ApiError(f"건수 불일치: 기대 {d.get('totCnt')}, 수신 {len(rows)}")
    return rows


def hotspots(url: str) -> pd.DataFrame:
    df = pd.DataFrame(fetch_all(url))
    out = pd.DataFrame({
        "city": df.sidoSigungu.str.contains("서귀포").map({True: "서귀포시", False: "제주시"}),
        "rank_in_city": df.sidoSigungu.str.extract(r"(\d+)$")[0].astype("Int64"),
        # 표기가 '제주특별자치도 ...' / '제주 ...' 로 섞여 있어 통일
        "place": df.occurrencePlace.str.replace(r"^제주특별자치도\s+|^제주\s+", "", regex=True).str.strip(),
        "place_raw": df.occurrencePlace,
        "bjd_code": df.bjdCode,
        "accidents": df.occurrenceCount,
        "casualties": df.deathInjuryCount,
        "deaths": df.deathCount,
        "serious": df.seriouslyInjuryCount,
        "slight": df.slightlyInjuryCount,
        "reported": df.injuryReporterCount,
        "lat": df.latitude.astype(float),
        "lon": df.longitude.astype(float),
    })
    # API 는 '시점별 도시당 상위 K곳' 순위표를 이어 붙인 형태라 연도 정보 없이 같은 순위가 반복된다.
    # 원래 순서를 보존하기 위해 도시·순위별 등장 횟수를 시점 번호(block, 0부터)로 기록한다.
    out["block"] = out.groupby(["city", "rank_in_city"]).cumcount()
    out["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    return out
