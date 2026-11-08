"""CSV 파서: 파일마다 다른 형식(컬럼명, 합계 행, 쉼표, 공백)을 공통 스키마로 정리한다.

공통 규칙
- 컬럼은 영문 snake_case, 문자열은 앞뒤 공백 제거, 숫자의 천단위 쉼표 제거
- 합계/소계 행은 지우지 않고 `level` 로 표시한다 (detail | sub_total | grand_total)
  -> 분석 시 level='detail' 만 합산하면 중복 집계가 생기지 않는다.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

NUM = {"사고 건수": "accidents", "부상자 수": "injured", "사망자 수": "deaths"}


def read_csv(path: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(path, encoding=enc, thousands=",")
            break
        except Exception as e:  # 인코딩 후보를 차례로 시도
            last = e
    else:
        raise last
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.select_dtypes(include="object"):
        df[c] = df[c].astype("string").str.strip()
    return df


def _ym(df, col):
    v = df[col].astype(int)
    df["ym"] = v.astype(str)
    df["year"] = v // 100
    df["month"] = v % 100
    return df.drop(columns=[col])


def _rename(df, extra: dict) -> pd.DataFrame:
    return df.rename(columns={**NUM, **extra})


# ----------------------------------------------------------------- 파서들
def intersection(df):
    df = _rename(df, {"교차로형태대분류": "major", "교차로형태소분류": "minor"})
    df = _ym(df, "기준 연월")
    df["level"] = "detail"
    df.loc[df.minor == "합계", "level"] = "sub_total"
    df.loc[df.major == "합계", "level"] = "grand_total"
    return df[["ym", "year", "month", "major", "minor", "level", "accidents", "injured", "deaths"]]


def weather(df):
    df = _rename(df, {"기준 연도": "year", "기상 상태": "weather"})
    return df[["year", "weather", "accidents", "injured", "deaths"]]


def surface(df):
    df = _rename(df, {"기준 연도": "year", "노면 상태": "surface"})
    return df[["year", "surface", "accidents", "injured", "deaths"]]


def elderly(df):
    df = _rename(df, {})
    df = _ym(df, "기준 연도")  # 이 파일은 '기준 연도' 컬럼에 YYYYMM 이 들어 있다
    return df[["ym", "year", "month", "accidents", "injured", "deaths"]]


def alignment(df):
    df = _rename(df, {"기준 연도": "year", "도로선형 대분류": "major", "도로선형 중분류": "mid", "도로선형 소분류": "minor"})
    df["level"] = "detail"
    df.loc[df.minor == "합계", "level"] = "sub_total"
    df.loc[(df.mid == "합계") & (df.minor == "합계"), "level"] = "major_total"
    return df[["year", "major", "mid", "minor", "level", "accidents", "injured", "deaths"]]


def child_road(df):
    df = _rename(df, {"기준 연도": "year", "도로 종류": "road_type"})
    df["level"] = df.road_type.eq("합계").map({True: "grand_total", False: "detail"})
    return df[["year", "road_type", "level", "accidents", "injured", "deaths"]]


BANDS = [(h, h + 2) for h in range(0, 24, 2)]


def child_hour(df):
    df = _rename(df, {"기준 연도": "year", "시간대": "band"})
    df["level"] = df.band.eq("합계").map({True: "grand_total", False: "detail"})
    m = df.band.str.extract(r"(\d+)시~(\d+)시")
    df["start_hour"], df["end_hour"] = m[0].astype("Int64"), m[1].astype("Int64")
    df["imputed"] = 0
    # 사고가 없는 시간대는 행 자체가 빠져 있다 -> 0건으로 채우고 표시(imputed=1)
    add = []
    for y, g in df[df.level == "detail"].groupby("year"):
        have = set(g.start_hour.dropna())
        for s, e in BANDS:
            if s not in have:
                add.append(dict(year=y, band=f"{s}시~{e}시", level="detail", start_hour=s, end_hour=e,
                                accidents=0, injured=0, deaths=0, imputed=1))
    if add:
        df = pd.concat([df, pd.DataFrame(add)], ignore_index=True)
    df = df.sort_values(["year", "start_hour"], na_position="first").reset_index(drop=True)
    return df[["year", "band", "start_hour", "end_hour", "level", "accidents", "injured", "deaths", "imputed"]]


_PED_COLS = {"발생일시": "date", "발생요일": "weekday", "시": "city", "읍면동": "dong", "사고내용": "severity", "사망자수": "deaths",
             "중상자수": "serious", "경상자수": "slight", "부상신고자": "reported", "사고유형": "acc_type", "법규위반": "violation",
             "노면상태": "surface", "기상상태": "weather", "도로형태": "road_shape", "피해연령대": "victim_age", "위도": "lat",
             "경도": "lon", "도로명": "road"}
# 피해자 연령대: 표본이 극히 적은 구분은 인접 구분에 합친다
_AGE_FIX = {"98세이상": "90대이상", "미분류": "미상"}


def ped_accidents(df):
    """보행자 사고(TAAS) - 한 행이 사고 1건. 시간·요일·기상·노면·위반·피해자·좌표가 모두 있어 요인 교차 분석과 지점 지도가 가능하다."""
    d = df.rename(columns=_PED_COLS)
    d["date"] = pd.to_datetime(d["date"])
    d["year"], d["month"] = d.date.dt.year, d.date.dt.month
    d["date"] = d.date.dt.strftime("%Y-%m-%d")
    # 발생시간이 '6:00' 형식. (원본의 '시' 컬럼은 시간이 아니라 행정시(제주시/서귀포시)다)
    d["hour"] = pd.to_numeric(d["발생시간"].astype(str).str.split(":").str[0], errors="coerce").astype("Int64")
    d["surface"] = d.surface.astype("string").str.split(" - ").str[-1]        # '포장 - 건조' -> '건조'
    d["road_main"] = d.road_shape.astype("string").str.split(" - ").str[0]    # '교차로 - 교차로부근' -> '교차로'
    d["victim_age"] = d.victim_age.replace(_AGE_FIX)
    d["geo_ok"] = (d.lat.between(33.0, 33.7) & d.lon.between(126.0, 127.1)).astype(int)  # 제주 범위 밖 좌표는 지도에서 제외
    d["acc_id"] = range(1, len(d) + 1)  # 원본 사고번호는 지수 표기로 저장되어 식별자로 쓸 수 없다
    cols = ["acc_id", "date", "year", "month", "weekday", "hour", "city", "dong", "severity", "deaths", "serious", "slight", "reported",
            "acc_type", "violation", "weather", "surface", "road_shape", "road_main", "victim_age", "lat", "lon", "road", "geo_ok"]
    return d[cols]


def enforcement(df):
    """법규위반별 교통단속(월). 가로로 펼쳐진 유형 컬럼을 (월, 유형, 건수)로 세운다."""
    d = df.rename(columns={"연도별": "date"})
    d["date"] = pd.to_datetime(d["date"])
    long = d.melt(id_vars="date", var_name="kind", value_name="count")
    long["ym"] = long.date.dt.strftime("%Y%m")
    long["year"], long["month"] = long.date.dt.year, long.date.dt.month
    return long[["ym", "year", "month", "kind", "count"]].sort_values(["ym", "kind"]).reset_index(drop=True)


# 파일명 키워드 -> (테이블, 파서, 설명). 위에서부터 먼저 맞는 것 사용.
REGISTRY = [
    ("보행자사고", "acc_ped", ped_accidents, "보행자 사고(TAAS, 사고 단위)"),
    ("법규위반별교통단속", "enf_monthly", enforcement, "법규위반별 교통단속(월)"),
    ("교차로", "acc_intersection", intersection, "교차로 형태별 사고(월)"),
    ("기상", "acc_weather", weather, "기상 상태별 사고(연)"),
    ("노면", "acc_surface", surface, "노면 상태별 사고(연)"),
    ("노인", "acc_elderly", elderly, "노인 교통사고(월)"),
    ("선형", "acc_alignment", alignment, "도로선형별 사고(연)"),
    ("도로종류별", "acc_child_road", child_road, "어린이 사고 - 도로종류(연)"),
    ("시간대별", "acc_child_hour", child_hour, "어린이 사고 - 시간대(연)"),
]


# 설문 기반 자료는 사용하지 않는다(실제 사고가 아닌 체감 응답이라 분석 대상에서 제외)
EXCLUDE = ("어린이교통사각지대", "무단횡단유발", "초등학교주변위험")


def excluded(path: Path) -> bool:
    return any(k in path.name for k in EXCLUDE)


def match(path: Path):
    for kw, table, fn, desc in REGISTRY:
        if kw in path.name:
            return table, fn, desc
    return None
