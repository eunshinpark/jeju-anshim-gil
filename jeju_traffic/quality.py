"""자동 품질 점검. 발견한 문제는 quality_issues 테이블에 남기고, 분석 단계(Agent)가 답변에 반영한다."""
from __future__ import annotations

import pandas as pd

from .config import HOTSPOT_PERIOD


def _issue(table, severity, code, message):
    return dict(table=table, severity=severity, code=code, message=message)


def _sum_check(df, keys, total_mask, detail_mask, table, label):
    """detail 합계 == 합계 행 인지 검사."""
    out = []
    d = df[detail_mask].groupby(keys)[["accidents", "injured", "deaths"]].sum()
    t = df[total_mask].groupby(keys)[["accidents", "injured", "deaths"]].sum()
    j = d.join(t, lsuffix="_detail", rsuffix="_total", how="inner")
    bad = j[(j.accidents_detail != j.accidents_total) | (j.injured_detail != j.injured_total)
            | (j.deaths_detail != j.deaths_total)]
    if len(bad):
        out.append(_issue(table, "warn", "TOTAL_MISMATCH",
                          f"{label}: 세부 합계와 합계 행이 다른 구간이 {len(bad)}곳 있습니다."))
    else:
        out.append(_issue(table, "info", "TOTAL_OK", f"{label}: 세부 합계와 합계 행이 일치합니다({len(j)}개 구간)."))
    return out


def check(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    issues: list[dict] = []
    T = tables

    # 1) 기상 파일 규모 이상 (제주 값이 아닌 구간이 섞였는지) - 같은 해 노면 파일과 비교
    if "acc_weather" in T and "acc_surface" in T:
        w = T["acc_weather"].groupby("year").accidents.sum()
        s = T["acc_surface"].groupby("year").accidents.sum()
        ratio = (w / s).dropna()
        bad_years = sorted(ratio[ratio > 3].index.tolist())
        T["acc_weather"]["suspect"] = T["acc_weather"].year.isin(bad_years).astype(int)
        if bad_years:
            issues.append(_issue(
                "acc_weather", "error", "SCALE_MISMATCH",
                f"기상 상태별 사고 건수가 같은 해 노면 상태별 건수의 3배를 넘는 해가 있습니다"
                f"({bad_years[0]}~{bad_years[-1]}년, {len(bad_years)}개 연도). "
                f"전국 규모 값이 섞인 것으로 보여 해당 연도는 분석에서 제외해야 합니다(suspect=1)."))
        else:
            issues.append(_issue("acc_weather", "info", "SCALE_OK", "기상/노면 파일의 연간 규모가 일치합니다."))

    # 2) 합계 행 검증
    if "acc_intersection" in T:
        df = T["acc_intersection"]
        issues += _sum_check(df, ["ym"], df.level == "grand_total", df.level == "detail",
                             "acc_intersection", "교차로 월별")
    if "acc_alignment" in T:
        df = T["acc_alignment"]
        issues += _sum_check(df, ["year", "major"], df.level == "major_total", df.level == "detail",
                             "acc_alignment", "도로선형 대분류별")
    for t, label in (("acc_child_road", "어린이 도로종류별"), ("acc_child_hour", "어린이 시간대별")):
        if t in T:
            df = T[t]
            issues += _sum_check(df, ["year"], df.level == "grand_total", df.level == "detail", t, label)

    # 3) 시간대 보정(사고 없는 시간대 행 누락)
    if "acc_child_hour" in T:
        n = int(T["acc_child_hour"].imputed.sum())
        if n:
            issues.append(_issue("acc_child_hour", "info", "IMPUTED_ZERO",
                                 f"사고가 없어 행이 빠진 시간대 {n}개를 0건으로 채웠습니다(imputed=1)."))

    # 4) 결측치
    for name, df in T.items():
        if name == "hotspots":
            continue
        skip = [c for c in ("suspect", "imputed", "start_hour", "end_hour") if c in df]  # 파생 컬럼은 제외
        n = int(df.drop(columns=skip).isna().any(axis=1).sum())
        if n:
            extra = ""
            if name == "acc_surface" and df[df.accidents.isna()].surface.nunique() == 1:
                extra = f" (모두 '{df[df.accidents.isna()].surface.iloc[0]}' 항목: 해당 사고가 없는 것으로 추정)"
            issues.append(_issue(name, "warn", "MISSING_VALUES", f"결측값이 있는 행이 {n}개 있습니다{extra}."))

    # 5) 기간(커버리지)
    for name, df in T.items():
        if "year" in df.columns:
            issues.append(_issue(name, "info", "COVERAGE", f"기간: {int(df.year.min())}~{int(df.year.max())}년"))

    # 7) 보행자 사고(사고 단위)
    if "acc_ped" in T:
        p = T["acc_ped"]
        bad = int((p.geo_ok == 0).sum())
        if bad:
            issues.append(_issue("acc_ped", "warn", "GEO_OUT", f"제주 범위를 벗어난 좌표가 {bad}건 있습니다."))
        dup = int(p.duplicated(["date", "hour", "lat", "lon", "severity"]).sum())
        if dup:
            issues.append(_issue("acc_ped", "info", "DUP_ROWS", f"날짜·시간·좌표·심각도가 같은 행이 {dup}건 있습니다(같은 사고의 복수 행일 수 있음)."))
        rare = p.weather.value_counts()
        small = [f"{k} {int(v)}건" for k, v in rare.items() if v < 30]
        if small:
            issues.append(_issue("acc_ped", "info", "SMALL_SAMPLE", "기상별 표본이 작은 구분이 있어 비교가 불안정합니다: " + ", ".join(small)))
        issues.append(_issue("acc_ped", "info", "COUNT", f"보행자 사고 {len(p):,}건 (사망자 {int(p.deaths.sum())}명), 지도에 쓸 수 있는 좌표 {int(p.geo_ok.sum()):,}건"))

    # 8) 법규위반 단속(월)
    if "enf_monthly" in T:
        e = T["enf_monthly"]
        m = e.groupby("ym")["count"].sum()
        per = pd.PeriodIndex(pd.to_datetime(m.index, format="%Y%m"), freq="M")
        gaps = int((pd.Series(per.asi8).diff().dropna() != 1).sum())
        if gaps:
            issues.append(_issue("enf_monthly", "warn", "MONTH_GAP", f"월별 기록에 빠진 달이 있습니다({gaps}곳)."))
        yr = e.groupby("year")["count"].sum()
        if len(yr) >= 4:
            med = float(yr.median())
            spikes = [f"{int(y)}년({v / med:.1f}배)" for y, v in yr.items() if v / med >= 1.6]
            if spikes:
                issues.append(_issue("enf_monthly", "warn", "SPIKE_YEAR",
                                     f"연간 단속 건수가 중앙값의 1.6배 이상인 해가 있습니다: {', '.join(spikes)}. 단속 방침 변화 등 원인 확인이 필요합니다."))

    # 6) 다발지역 API
    if "hotspots" in T:
        h = T["hotspots"]
        out = h[(h.lat < 33.0) | (h.lat > 33.7) | (h.lon < 126.0) | (h.lon > 127.1)]
        if len(out):
            issues.append(_issue("hotspots", "error", "COORD_OUT_OF_JEJU", f"제주 범위를 벗어난 좌표 {len(out)}건"))
        calc = h.deaths + h.serious + h.slight + h.reported
        mism = int((calc != h.casualties).sum())
        issues.append(_issue("hotspots", "warn" if mism else "info", "CASUALTY_SUM",
                             f"사상자수 = 사망+중상+경상+부상신고 검산: 불일치 {mism}건"))
        per = h.groupby(["city", "rank_in_city"]).size()
        if per.nunique() == 1 and per.iloc[0] > 1:
            k = h.rank_in_city.max()
            issues.append(_issue(
                "hotspots", "warn", "REPEATED_RANKS",
                f"같은 순위가 {per.iloc[0]}번씩 반복됩니다(도시당 상위 {k}곳 × {per.iloc[0]}개 시점). "
                f"제공처 설명상 기간이 {HOTSPOT_PERIOD}이고 순위표가 {per.iloc[0]}개라 연도별 순위표로 보이지만, "
                f"API가 연도 조건을 지원하지 않아 시점(block 0~{per.iloc[0]-1})과 연도의 대응은 확인할 수 없고, "
                f"장소 중복 {int(h.duplicated(['place']).sum())}건은 오류가 아니라 여러 시점에 반복 등장한 것입니다. "
                f"고유 장소는 {h.place.nunique()}곳입니다."))
        else:
            dup = int(h.duplicated(["place"]).sum())
            if dup:
                issues.append(_issue("hotspots", "warn", "DUP_PLACE", f"장소명이 중복된 행 {dup}건"))
        issues.append(_issue("hotspots", "warn", "NO_REFERENCE_YEAR",
                             "다발지역 API 응답에는 연도 필드가 없고 연도 조회 조건도 지원하지 않습니다"
                             f"(제공처 설명: {HOTSPOT_PERIOD}). 시점별 연도는 단정하지 않습니다."))
        issues.append(_issue("hotspots", "info", "COUNT", f"{len(h)}곳 (제주시 {int((h.city=='제주시').sum())}, "
                                                          f"서귀포시 {int((h.city=='서귀포시').sum())})"))
    return T, issues
