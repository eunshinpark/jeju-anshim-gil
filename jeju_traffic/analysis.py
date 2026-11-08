"""분석 도구. 모든 숫자는 SQL/pandas 로 계산하고, LLM 은 결과를 해석·설명만 한다.

Result.summary 에는 근거 수치와 한계(※)를 함께 담아, 어떤 경로로 답하더라도 주의사항이 빠지지 않게 한다.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import HOTSPOT_PERIOD, db_path

CAVEAT = ("※ 사고 건수는 해당 도로·환경의 교통량(노출도)을 반영하지 않아, 건수가 많다고 더 위험하다는 뜻은 아닙니다. "
          "치명도는 사고 100건당 사망자 수입니다.")


@dataclass
class Result:
    title: str
    summary: str
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    charts: list[dict] = field(default_factory=list)  # {"kind": bar|line, "x":, "y":, "horizontal":}
    map_df: pd.DataFrame | None = None
    years: tuple[int, int] | None = None  # 분석에 쓴 연도 범위(문구용)
    meta: dict = field(default_factory=dict)  # 도구별 부가 수치(보고서용)
    map_legend: str = ""  # 지도 범례 문구


def _q(sql: str, params=()) -> pd.DataFrame:
    p = db_path()
    if not p.exists():
        raise FileNotFoundError("DB가 없습니다. 먼저 `python -m jeju_traffic.collect` 를 실행하세요.")
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    try:
        return pd.read_sql(sql, con, params=params)
    finally:
        con.close()


def _has(table: str) -> bool:
    try:
        return len(_q("select name from sqlite_master where type='table' and name=?", (table,))) > 0
    except FileNotFoundError:
        return False


def _filter_years(df, start, end):
    if start:
        df = df[df.year >= int(start)]
    if end:
        df = df[df.year <= int(end)]
    return df


def _metrics(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["치명도"] = (g.deaths / g.accidents * 100).round(2)
    g["건당부상"] = (g.injured / g.accidents).round(2)
    g["구성비(%)"] = (g.accidents / g.accidents.sum() * 100).round(1)
    return g


# ------------------------------------------------------------- 1. 요인별 위험도
_FACTORS = {"intersection": "교차로 형태별", "alignment": "도로선형별", "weather": "기상 상태별", "surface": "노면 상태별"}


def risk_by_factor(factor: str = "intersection", start_year=None, end_year=None) -> Result:
    factor = factor if factor in _FACTORS else "intersection"
    note = ""
    if factor == "intersection":
        df = _q("select year, minor label, accidents, injured, deaths from acc_intersection where level='detail'")
        df["label"] = df.label.str.replace("교차로 - ", "", regex=False).replace({"교차로아님": "교차로 아님"})
    elif factor == "alignment":
        df = _q("select year, major, mid, minor, accidents, injured, deaths from acc_alignment where level='detail'")
        df["label"] = [f"{a} {b} {c}".replace("직선 직선", "직선").replace("기타구역 기타구역", "기타구역").strip()
                       for a, b, c in zip(df.major, df.mid, df.minor)]
        df = df.drop(columns=["major", "mid", "minor"])
    elif factor == "weather":
        df = _q("select year, weather label, accidents, injured, deaths from acc_weather where suspect=0")
        bad = _q("select min(year) a, max(year) b from acc_weather where suspect=1").iloc[0]
        note = "" if pd.isna(bad.a) else f" ({int(bad.a)}~{int(bad.b)}년 값은 전국 규모로 의심되어 제외)"
    else:
        df = _q("select year, surface label, accidents, injured, deaths from acc_surface")
    df = _filter_years(df, start_year, end_year)
    df = df.dropna(subset=["accidents"])
    if df.empty:
        return Result(_FACTORS[factor], "해당 기간의 자료가 없습니다. 기간을 바꿔 보세요.")
    y0, y1 = int(df.year.min()), int(df.year.max())
    g = _metrics(df.groupby("label")[["accidents", "injured", "deaths"]].sum()).sort_values("accidents", ascending=False)
    g = g.reset_index().rename(columns={"label": "구분", "accidents": "사고 건수", "injured": "부상자", "deaths": "사망자"})
    top = g.iloc[0]
    # 의미 없는 분류(기타/불명)와 표본이 작은(30건 미만) 구분은 치명도 순위에서 제외
    big = g[(g["사고 건수"] >= 30) & ~g.구분.isin(["기타/불명", "기타"])]
    lethal = big.sort_values("치명도", ascending=False).iloc[0] if len(big) else None
    s = (f"{y0}~{y1}년{note} {_FACTORS[factor]} 사고입니다. 사고가 가장 많은 구분은 '{top.구분}'"
         f"({int(top['사고 건수']):,}건, {top['구성비(%)']}%)입니다.")
    if lethal is not None:
        s += (f" 사고 30건 이상인 구분 중 치명도가 가장 높은 곳은 '{lethal.구분}'"
              f"(사고 100건당 사망 {lethal.치명도}명)입니다.")
    base = {"weather": "맑음", "surface": "건조"}.get(factor)
    if base and base in set(g.구분):
        b = g[g.구분 == base].iloc[0]
        cmp_ = []
        for r in big[big.구분 != base].itertuples():
            if b.치명도 > 0:
                cmp_.append(f"'{r.구분}' {r.치명도}명({r.치명도 / b.치명도:.1f}배)")
        if cmp_:
            s += (f" 기준 상태({base})의 치명도는 100건당 사망 {b.치명도}명이고, 이에 견주면 " + ", ".join(cmp_) + "입니다.")
        s += " 사고 수가 적은 구분은 치명도가 크게 흔들릴 수 있어 참고용으로 보세요."
    dim = g[(g["사고 건수"] < 30) | g.구분.isin(["기타/불명", "기타"])].구분.tolist()  # 차트에서 회색 처리
    return Result(f"{_FACTORS[factor]} 사고 현황", s + " " + CAVEAT, g,
                  [{"kind": "bar", "x": "구분", "y": "사고 건수", "dim": dim},
                   {"kind": "bar", "x": "구분", "y": "치명도", "dim": dim}], years=(y0, y1))


# ------------------------------------------------------------- 2. 월별 추이 (노인·교차로 전체)
_TREND = {"elderly": "노인 교통사고", "intersection": "전체 교통사고(교차로 파일 합계)"}


def monthly_trend(target: str = "elderly", start_year=None, end_year=None) -> Result:
    target = target if target in _TREND else "elderly"
    if target == "elderly":
        df = _q("select ym, year, month, accidents, deaths, injured from acc_elderly order by ym")
    else:
        df = _q("select ym, year, month, accidents, deaths, injured from acc_intersection where level='grand_total' order by ym")
    df = _filter_years(df, start_year, end_year)
    if df.empty:
        return Result(_TREND[target], "해당 기간의 자료가 없습니다.")
    df["월"] = df.ym.str[:4] + "-" + df.ym.str[4:]
    yr = df.groupby("year")[["accidents", "deaths", "injured"]].sum()
    mo = df.groupby("month").accidents.mean().round(1)
    peak, low = mo.idxmax(), mo.idxmin()
    first, last = yr.index.min(), yr.index.max()
    chg = (yr.accidents[last] / yr.accidents[first] - 1) * 100 if first != last else 0
    s = (f"{_TREND[target]}의 월별 추이입니다({df.ym.iloc[0][:4]}-{df.ym.iloc[0][4:]} ~ {df.ym.iloc[-1][:4]}-{df.ym.iloc[-1][4:]}). "
         f"연간 사고는 {first}년 {int(yr.accidents[first]):,}건에서 {last}년 {int(yr.accidents[last]):,}건으로 "
         f"{abs(chg):.0f}% {'증가' if chg > 0 else '감소'}했습니다. 월평균으로 가장 많은 달은 {peak}월({mo[peak]}건), "
         f"가장 적은 달은 {low}월({mo[low]}건)입니다. ※ 관찰 기간이 {last-first+1}년이라 계절성은 참고용입니다.")
    out = df.rename(columns={"accidents": "사고 건수", "deaths": "사망자", "injured": "부상자"})[["월", "사고 건수", "사망자", "부상자"]]
    return Result(f"{_TREND[target]} 월별 추이", s, out, [{"kind": "line", "x": "월", "y": "사고 건수"}],
                  years=(int(first), int(last)))


# ------------------------------------------------------------- 2-1. 월별 전망
MIN_TRAIN, MIN_TOTAL, MAX_MAPE = 24, 30, 0.35  # 학습 24개월 + 검증 6개월 이상 있어야 전망한다
_FC_TARGETS = {**_TREND, "enforcement": "법규위반 단속 건수"}
FC_CAVEAT = ("※ 참고용 전망입니다. 보유 기간의 계절 패턴과 최근 12개월 수준이 이어진다고 가정했으므로 증가·감소 추세는 반영하지 않았고, "
             "날씨·교통량·정책 변화도 반영하지 않았습니다. 음영은 80% 범위이며 검증 구간의 오차에서 계산했습니다.")


def _season_fc(y: np.ndarray, h: int) -> np.ndarray:
    """계절지수 × 최근 12개월 수준. 완결된 연도의 월별 비율로 계절지수를 만든다."""
    n = len(y)
    yrs = n // 12
    block = y[: yrs * 12].reshape(yrs, 12)
    idx = (block / block.mean(axis=1, keepdims=True)).mean(axis=0)
    idx = idx / idx.mean()
    level = y[-12:].mean()
    return np.array([level * idx[(n + i) % 12] for i in range(h)])


def _naive_fc(y: np.ndarray, h: int) -> np.ndarray:
    """단순 기준선: 작년 같은 달 값."""
    n = len(y)
    return np.array([y[n - 12 + (i % 12)] for i in range(h)])


def _backtest(y: np.ndarray, h: int = 12):
    """롤링 검증: 학습 구간을 늘려 가며 이후 최대 h개월을 예측해 오차를 모은다.
    반환: (실제/예측-1 배열, 모델 MAPE, 단순 기준선 MAPE)"""
    rel, ape, ape_n = [], [], []
    for t in range(MIN_TRAIN, len(y)):
        H = min(h, len(y) - t)
        a, f, fn = y[t:t + H], _season_fc(y[:t], H), _naive_fc(y[:t], H)
        ok = a > 0
        rel += list(a[ok] / f[ok] - 1)
        ape += list(np.abs(a[ok] - f[ok]) / a[ok])
        ape_n += list(np.abs(a[ok] - fn[ok]) / a[ok])
    return np.array(rel), float(np.mean(ape)), float(np.mean(ape_n))


def forecast(target: str = "elderly", horizon: int = 12) -> Result:
    if target not in _FC_TARGETS:
        return Result("전망", "이 항목은 월별 기록이 없어 전망할 수 없습니다. 전망할 수 있는 항목은 노인 교통사고와 전체 교통사고(월별 기록이 있는 자료)입니다.")
    horizon = max(1, min(int(horizon or 12), 12))
    if target == "elderly":
        df = _q("select ym, accidents a from acc_elderly order by ym")
    elif target == "enforcement":
        if not _has("enf_monthly"):
            return Result("전망", "단속 데이터가 아직 없습니다. 수집을 먼저 실행하세요.")
        df = _q("select ym, sum(count) a from enf_monthly group by ym order by ym")
    else:
        df = _q("select ym, accidents a from acc_intersection where level='grand_total' order by ym")
    label = _FC_TARGETS[target]
    val = "단속 건수" if target == "enforcement" else "사고 건수"
    y = df.a.to_numpy(float)
    per = pd.to_datetime(df.ym, format="%Y%m").dt.to_period("M")
    if len(y) < MIN_TOTAL:
        return Result("전망", f"{label}의 월별 기록이 {len(y)}개월뿐이라 전망하지 않습니다(학습 {MIN_TRAIN}개월 + 검증 6개월, 최소 {MIN_TOTAL}개월 필요).")
    if not (np.diff(per.astype("int64")) == 1).all():
        return Result("전망", f"{label}의 월별 기록에 빠진 달이 있어 전망하지 않습니다.")

    f = _season_fc(y, horizon)
    rel, mape, mape_n = _backtest(y)
    if mape > MAX_MAPE:  # 검증 오차가 너무 크면 전망하지 않는다(급격한 변화 구간 등으로 '이어진다'는 가정이 맞지 않음)
        return Result("전망", f"{label}: 검증에서 월별 평균 오차가 {mape * 100:.0f}%(단순 방법 {mape_n * 100:.0f}%)로 커서 전망을 제공하지 않습니다. "
                      "자료에 급격한 변화가 있는 구간이 있으면 계절 패턴과 최근 수준이 이어진다는 가정이 맞지 않기 때문입니다.")
    if len(rel) >= 30:
        q10, q90 = np.quantile(rel, [0.1, 0.9])
    else:
        s = float(rel.std())
        q10, q90 = -1.28 * s, 1.28 * s
    lo, hi = np.maximum(f * (1 + q10), 0), f * (1 + q90)

    last = per.iloc[-1]
    fut_m = [(last + i + 1).strftime("%Y-%m") for i in range(horizon)]
    out = pd.concat([
        pd.DataFrame({"월": per.dt.strftime("%Y-%m"), "구분": "실적", val: y}),
        pd.DataFrame({"월": fut_m, "구분": "전망", val: np.round(f), "하한": np.round(lo), "상한": np.round(hi)})],
        ignore_index=True)

    tot, last12 = float(f.sum()), float(y[-12:].sum())
    ip, il = int(f.argmax()), int(f.argmin())
    peak, low = int(fut_m[ip][-2:]), int(fut_m[il][-2:])
    diff = (mape_n - mape) * 100
    verdict = (f"단순 방법보다 {diff:.1f}%p 나았습니다" if diff >= 1 else
               f"단순 방법보다 {abs(diff):.1f}%p 나빴습니다" if diff <= -1 else "단순 방법과 비슷한 수준입니다")
    # 이 방법은 최근 12개월 수준이 유지된다고 가정하므로 12개월 합계는 최근 12개월과 같고, 달라지는 것은 월별 분포뿐이다.
    if horizon == 12:
        core = (f"최근 12개월 수준({last12:,.0f}건)이 이어진다고 가정하면 합계는 같은 수준이고, 달별로는 {peak}월(약 {f[ip]:,.0f}건)이 "
                f"가장 많고 {low}월(약 {f[il]:,.0f}건)이 가장 적을 것으로 전망됩니다.")
    else:
        core = f"월평균 약 {f.mean():,.0f}건이며, {peak}월(약 {f[ip]:,.0f}건)이 가장 많을 것으로 전망됩니다."
    s = (f"{label}의 향후 {horizon}개월({fut_m[0]} ~ {fut_m[-1]}) 전망입니다. {core} 보유 기간의 마지막 구간을 가려 검증했을 때 "
         f"월별 평균 오차는 {mape * 100:.0f}%였고, '작년 같은 달' 단순 방법은 {mape_n * 100:.0f}%였습니다({verdict}). " + FC_CAVEAT)
    return Result(f"{label} 향후 {horizon}개월 전망", s, out, [{"kind": "forecast", "x": "월", "y": val}],
                  years=(int(per.iloc[0].year), int(last.year)),
                  meta=dict(horizon=horizon, total=tot, last12=last12, m0=fut_m[0], m1=fut_m[-1], mape=mape, mape_naive=mape_n,
                            n_months=len(y), val=val, peak=peak, peak_v=float(f[ip]), low=low, low_v=float(f[il]), avg=float(f.mean())))


# ------------------------------------------------------------- 3. 어린이 사고
def child_risk(by: str = "hour", start_year=None, end_year=None) -> Result:
    if by == "road":
        df = _filter_years(_q("select year, road_type label, accidents, injured, deaths from acc_child_road where level='detail'"),
                           start_year, end_year)
        what = "도로 종류별"
    else:
        df = _filter_years(_q("select year, band label, start_hour, accidents, injured, deaths from acc_child_hour where level='detail'"),
                           start_year, end_year)
        what = "시간대별"
    if df.empty:
        return Result("어린이 교통사고", "해당 기간의 자료가 없습니다.")
    y0, y1 = int(df.year.min()), int(df.year.max())
    if by == "road":
        g = df.groupby("label")[["accidents", "injured", "deaths"]].sum()
        g = _metrics(g).sort_values("accidents", ascending=False)
    else:
        g = df.groupby(["start_hour", "label"])[["accidents", "injured", "deaths"]].sum().reset_index().sort_values("start_hour")
        g = _metrics(g.set_index("label").drop(columns="start_hour"))
    g = g.reset_index().rename(columns={"label": "구분", "accidents": "사고 건수", "injured": "부상자", "deaths": "사망자"})
    top = g.sort_values("사고 건수", ascending=False).iloc[0]
    s = (f"{y0}~{y1}년 12세 이하 어린이 교통사고를 {what} 본 결과, 가장 많은 구분은 '{top.구분}'"
         f"({int(top['사고 건수']):,}건, {top['구성비(%)']}%)입니다.")
    if by == "hour":
        top3 = g.sort_values("사고 건수", ascending=False).head(3)
        s += f" 상위 3개 시간대({', '.join(top3.구분)})가 전체의 {top3['구성비(%)'].sum():.0f}%를 차지합니다."
    s += " ※ 사고 건수만 비교한 것이며, 같은 시간대·도로의 어린이 통행량은 반영되지 않았습니다."
    return Result(f"어린이 교통사고 {what}", s, g, [{"kind": "bar", "x": "구분", "y": "사고 건수"}], years=(y0, y1))


# ------------------------------------------------------------- 4. 다발지역
def hotspot_rank(city: str | None = None, n: int = 10) -> Result:
    if not _has("hotspots"):
        return Result("사고 다발지역", "다발지역 데이터가 아직 없습니다. 수집을 먼저 실행하세요.")
    h = _q("select * from hotspots")
    _k, _nb = int(h.rank_in_city.max()), int(h.block.max()) + 1  # 도시당 상위 K곳 × NB개 순위표
    if city:
        h = h[h.city == ("서귀포시" if "서귀" in city else "제주시")]
    if h.empty:
        return Result("사고 다발지역", "조건에 맞는 지역이 없습니다.")
    n = max(1, min(int(n or 10), 30))
    g = (h.groupby("place")
         .agg(등장횟수=("block", "size"), 평균사고=("accidents", "mean"), 최대사고=("accidents", "max"),
              평균사상자=("casualties", "mean"), lat=("lat", "mean"), lon=("lon", "mean"), 도시=("city", "first"))
         .reset_index().rename(columns={"place": "장소"}))
    g[["평균사고", "평균사상자"]] = g[["평균사고", "평균사상자"]].round(1)
    g = g.sort_values(["등장횟수", "평균사고"], ascending=False).head(n)
    top = g.iloc[0]
    where = f"{city} " if city else "제주 전체 "
    s = (f"{where}사고 다발지역입니다. 순위표에 가장 자주 등장하는 곳은 '{top.장소}'로 "
         f"{int(top.등장횟수)}개 시점에 올랐고 시점당 평균 {top.평균사고}건입니다. "
         f"※ 이 자료는 제공처 설명상 {HOTSPOT_PERIOD}의 도시별 상위 {_k}곳 순위표 {_nb}개를 합친 것으로 보이지만, "
         f"각 순위표가 몇 년도인지는 확인되지 않아 연도별 비교는 하지 않았습니다. "
         f"'등장 횟수'가 많을수록 여러 해에 걸쳐 반복해서 사고가 많은 곳으로 해석할 수 있습니다.")
    out = g[["장소", "도시", "등장횟수", "평균사고", "최대사고", "평균사상자"]]
    m = g[["lat", "lon", "장소", "도시", "등장횟수", "평균사고", "최대사고", "평균사상자"]].copy()
    m["값"], m["크기"] = m["등장횟수"], m["평균사고"]
    m["설명"] = [f"순위표 등장 {int(r.등장횟수)}회\n시점당 평균 사고 {r.평균사고}건 (최대 {int(r.최대사고)}건)\n평균 사상자 {r.평균사상자}명" for r in m.itertuples()]
    return Result("사고 다발지역", s, out, [{"kind": "bar", "x": "장소", "y": "평균사고", "horizontal": True}], m,
                  map_legend="원 크기 = 시점당 평균 사고 건수 · 숫자 = 순위표 등장 횟수")


# ------------------------------------------------------------- 5. 데이터 현황·품질
def data_overview() -> Result:
    meta = _q("select table_name 테이블, description 설명, source 출처파일, rows 행수 from source_files")
    per = []
    for t in meta.테이블:
        cols = [r[1] for r in sqlite3.connect(db_path()).execute(f"pragma table_info({t})").fetchall()]
        if "year" in cols:
            r = _q(f"select min(year) a, max(year) b from {t}").iloc[0]
            per.append(f"{int(r.a)}~{int(r.b)}년")
        elif t == "hotspots":
            per.append(f"{HOTSPOT_PERIOD}(제공처 설명, 연도별 구분 불가)")
        else:
            per.append("연도 정보 없음")
    meta["기간"] = per
    iss = _q("select severity, count(*) n from quality_issues group by severity").set_index("severity").n.to_dict()
    s = (f"교통사고 관련 데이터 {len(meta)}종을 보유하고 있습니다. "
         f"자동 점검 결과: 오류 {iss.get('error', 0)}건, 주의 {iss.get('warn', 0)}건, 참고 {iss.get('info', 0)}건. "
         f"자세한 내용은 '데이터 품질 점검'을 요청하세요.")
    return Result("보유 데이터 현황", s, meta)


def quality_report() -> Result:
    df = _q("select severity 등급, table_name 테이블, code 코드, message 내용 from "
            "(select severity, \"table\" table_name, code, message from quality_issues) "
            "order by case severity when 'error' then 0 when 'warn' then 1 else 2 end")
    e = df[df.등급.isin(["error", "warn"])]
    s = "자동 점검에서 발견된 문제입니다. " + (" / ".join(f"[{r.테이블}] {r.내용}" for r in e.itertuples())
                                     if len(e) else "발견된 문제가 없습니다.")
    return Result("데이터 품질 점검", s, df)


def kpis() -> list[dict]:
    """화면 상단 핵심 수치 카드(최신 연도 기준, 전년 대비)."""
    inter = _q("select year, sum(accidents) a, sum(deaths) d from acc_intersection where level='grand_total' group by year").set_index("year")
    eld = _q("select year, sum(accidents) a from acc_elderly group by year").set_index("year")
    kid = _q("select year, accidents a from acc_child_road where level='grand_total'").set_index("year")
    y = int(inter.index.max())

    def card(label, s, unit, icon):
        cur, prev = s.get(y), s.get(y - 1)
        sub = "" if prev in (None, 0) else (
            f"전년 대비 {'▲' if cur > prev else '▼' if cur < prev else '–'} {abs(cur / prev - 1) * 100:.1f}%")
        return dict(label=f"{y}년 {label}", value=f"{int(cur):,}", unit=unit, sub=sub, icon=icon)

    return [card("전체 교통사고", inter.a, "건", "car_crash"), card("사망자", inter.d, "명", "personal_injury"),
            card("노인 교통사고", eld.a, "건", "elderly"), card("어린이 교통사고", kid.a, "건", "child_care")]


# ------------------------------------------------------------- 보행자 사고 (TAAS: 사고 단위 자료)
_PED_BY = {"hour": "시간대", "weekday": "요일", "weather": "기상", "surface": "노면", "violation": "법규위반",
           "victim_age": "피해자 연령대", "road_shape": "도로 형태", "city": "지역(시)"}
_WEEK = ["월", "화", "수", "목", "금", "토", "일"]
_AGES = ["7세이하", "초등학생", "중학생", "고등학생", "20대", "30대", "40대", "50대", "60대", "70대", "80대", "90대이상", "미상"]
_AGE_GROUP = {"7세이하": "어린이·학생", "초등학생": "어린이·학생", "중학생": "어린이·학생", "고등학생": "어린이·학생",
              "20대": "20~30대", "30대": "20~30대", "40대": "40~50대", "50대": "40~50대",
              "60대": "60대 이상", "70대": "60대 이상", "80대": "60대 이상", "90대이상": "60대 이상"}
_PERIODS = [(0, 6, "새벽(0~6시)"), (6, 10, "아침(6~10시)"), (10, 16, "낮(10~16시)"), (16, 20, "저녁(16~20시)"), (20, 24, "밤(20~24시)")]
_ORDINAL = ("hour", "weekday", "victim_age")
PED_CAVEAT = "※ 보행자 사고 건수는 보행량·교통량을 반영하지 않아, 건수가 많다고 더 위험하다는 뜻은 아닙니다. 치명도는 사고 100건당 사망자이며, 사고 수가 적은 구분은 크게 흔들릴 수 있습니다."


def _ped(city=None, victim="all") -> pd.DataFrame:
    if not _has("acc_ped"):
        return pd.DataFrame()
    d = _q("select * from acc_ped")
    if city:
        d = d[d.city == ("서귀포시" if "서귀" in city else "제주시")]
    if victim == "child":
        d = d[d.victim_age.isin(["7세이하", "초등학생"])]  # 어린이: 7세 이하 + 초등학생
    elif victim == "elderly":
        d = d[d.victim_age.isin(["60대", "70대", "80대", "90대이상"])]
    return d


def _ped_label(d: pd.DataFrame, by: str, coarse: bool = False) -> pd.Series:
    if by == "hour":
        h = d.hour.astype(float)
        if coarse:
            lab = pd.Series(index=d.index, dtype=object)
            for a, b, name in _PERIODS:
                lab[(h >= a) & (h < b)] = name
            return lab
        return h.map(lambda v: None if pd.isna(v) else f"{int(v) // 2 * 2:02d}~{int(v) // 2 * 2 + 2:02d}시")
    if by == "victim_age" and coarse:
        return d.victim_age.map(_AGE_GROUP)
    col = {"weekday": "weekday", "weather": "weather", "surface": "surface", "violation": "violation",
           "victim_age": "victim_age", "road_shape": "road_main", "city": "city"}[by]
    return d[col]


def _ped_sorted(labels, by: str, coarse: bool = False) -> list:
    labs = [x for x in pd.unique(pd.Series(list(labels))) if pd.notna(x)]
    if by == "hour":
        return sorted(labs, key=(lambda s: [n for _, _, n in _PERIODS].index(s)) if coarse else (lambda s: int(s[:2])))
    if by == "weekday":
        return [w for w in _WEEK if w in labs]
    if by == "victim_age":
        order = ["어린이·학생", "20~30대", "40~50대", "60대 이상"] if coarse else _AGES
        return [a for a in order if a in labs]
    return labs  # 순서가 없는 요인은 건수순(호출 쪽에서 정렬)


def pedestrian_risk(ped_by: str = "hour") -> Result:
    d = _ped()
    if d.empty:
        return Result("보행자 사고", "보행자 사고 데이터가 아직 없습니다. 수집을 먼저 실행하세요.")
    ped_by = ped_by if ped_by in _PED_BY else "hour"
    d = d.assign(label=_ped_label(d, ped_by)).dropna(subset=["label"])
    g = d.groupby("label").agg(accidents=("acc_id", "size"), deaths=("deaths", "sum"), serious=("serious", "sum")).reset_index()
    g["치명도"] = (g.deaths / g.accidents * 100).round(2)
    g["구성비(%)"] = (g.accidents / g.accidents.sum() * 100).round(1)
    if ped_by in _ORDINAL:
        g = g.set_index("label").loc[_ped_sorted(g.label, ped_by)].reset_index()
    else:
        g = g.sort_values("accidents", ascending=False)
    g = g.rename(columns={"label": "구분", "accidents": "사고 건수", "deaths": "사망자", "serious": "중상자"})
    y0, y1 = int(d.year.min()), int(d.year.max())
    top = g.sort_values("사고 건수", ascending=False).iloc[0]
    big = g[(g["사고 건수"] >= 30) & ~g.구분.isin(["기타", "미상"])]
    lethal = big.sort_values("치명도", ascending=False).iloc[0] if len(big) else None
    s = (f"{y0}~{y1}년 보행자 사고 {len(d):,}건을 {_PED_BY[ped_by]}별로 본 결과, 가장 많은 구분은 '{top.구분}'"
         f"({int(top['사고 건수']):,}건, {top['구성비(%)']}%)입니다.")
    if ped_by == "hour":
        t3 = g.sort_values("사고 건수", ascending=False).head(3)
        s += f" 상위 3개 시간대({', '.join(t3.구분)})가 전체의 {t3['구성비(%)'].sum():.0f}%를 차지합니다."
    if lethal is not None:
        s += f" 사고 30건 이상인 구분 중 치명도가 가장 높은 곳은 '{lethal.구분}'(사고 100건당 사망 {lethal.치명도}명)입니다."
    dim = g[(g["사고 건수"] < 30) | g.구분.isin(["기타", "미상"])].구분.tolist()
    hz = ped_by in ("violation", "road_shape")
    return Result(f"보행자 사고 {_PED_BY[ped_by]}별", s + " " + PED_CAVEAT, g,
                  [{"kind": "bar", "x": "구분", "y": "사고 건수", "dim": dim, "horizontal": hz},
                   {"kind": "bar", "x": "구분", "y": "치명도", "dim": dim, "horizontal": hz}], years=(y0, y1))


def pedestrian_cross(rows: str = "weather", cols: str = "hour") -> Result:
    """보행자 사고를 두 요인으로 교차 집계(요인별 표가 따로 집계된 자료에서는 불가능했던 분석)."""
    d = _ped()
    if d.empty:
        return Result("보행자 사고 교차 분석", "보행자 사고 데이터가 아직 없습니다. 수집을 먼저 실행하세요.")
    rows = rows if rows in _PED_BY else "weather"
    cols = cols if cols in _PED_BY else "hour"
    if rows == cols:
        return pedestrian_risk(rows)
    d = d.assign(r=_ped_label(d, rows, True), c=_ped_label(d, cols, True)).dropna(subset=["r", "c"])

    def keep(col, by):  # 순서 있는 요인은 정해진 순서, 그 밖은 건수 상위 6개
        labs = d.groupby(col).size().sort_values(ascending=False).index.tolist()
        return _ped_sorted(labs, by, True) if by in _ORDINAL else labs[:6]

    ro, co = keep("r", rows), keep("c", cols)
    d = d[d.r.isin(ro) & d.c.isin(co)]
    t = d.groupby(["r", "c"]).agg(n=("acc_id", "size"), deaths=("deaths", "sum")).reset_index()
    full = pd.MultiIndex.from_product([ro, co], names=["r", "c"]).to_frame(index=False).merge(t, how="left", on=["r", "c"]).fillna(0)
    full["n"], full["deaths"] = full.n.astype(int), full.deaths.astype(int)
    full["치명도"] = (full.deaths / full.n.where(full.n > 0) * 100).round(1)
    rn, cn = _PED_BY[rows], _PED_BY[cols]
    out = full.rename(columns={"r": rn, "c": cn, "n": "사고 건수", "deaths": "사망자"})[[rn, cn, "사고 건수", "사망자", "치명도"]]
    y0, y1 = int(d.year.min()), int(d.year.max())
    top = out.sort_values("사고 건수", ascending=False).iloc[0]
    cells, small = len(out), float((out["사고 건수"] < 10).mean())
    s = (f"{y0}~{y1}년 보행자 사고 {int(out['사고 건수'].sum()):,}건을 {rn} × {cn} 조합으로 나눠 본 결과, 가장 많은 조합은 "
         f"'{top[rn]} × {top[cn]}'({int(top['사고 건수'])}건)입니다.")
    solid = out[out["사고 건수"] >= 20].sort_values("치명도", ascending=False)
    if len(solid):
        t1 = solid.iloc[0]
        s += f" 사고 20건 이상인 조합 중 치명도가 가장 높은 곳은 '{t1[rn]} × {t1[cn]}'(100건당 사망 {t1['치명도']}명)입니다."
    s += (f" 표본을 {cells}칸으로 나누어 칸의 {small * 100:.0f}%가 10건 미만이므로, 적은 칸의 차이는 우연일 수 있습니다. " + PED_CAVEAT)
    return Result(f"보행자 사고 {rn} × {cn}", s, out, [{"kind": "heat", "x": cn, "y": rn, "z": "사고 건수"}], years=(y0, y1))


def pedestrian_hotspots(city: str | None = None, n: int = 15, victim: str = "all") -> Result:
    """보행자 사고가 실제로 몰린 지점(약 400m 격자). 다발지역 API(42곳)와 달리 사고 좌표에서 직접 계산한다."""
    d = _ped(city, victim)
    if d.empty:
        return Result("보행자 사고 다발 지점", "보행자 사고 데이터가 아직 없습니다. 수집을 먼저 실행하세요.")
    d = d[d.geo_ok == 1]
    n = max(1, min(int(n or 15), 30))
    step = 0.004  # 위도 기준 약 440m
    d = d.assign(gy=(d.lat / step).round().astype(int), gx=(d.lon / step).round().astype(int))
    mode = lambda s: (s.dropna().mode().iloc[0] if s.notna().any() else "")
    rows = [dict(dong=mode(g.dong), road=mode(g.road), vio=mode(g.violation), n=len(g), deaths=int(g.deaths.sum()),
                 serious=int(g.serious.sum()), lat=float(g.lat.mean()), lon=float(g.lon.mean()))
            for _, g in d.groupby(["gy", "gx"])]
    t = pd.DataFrame(rows)
    t = t[t.n >= 2].sort_values(["n", "deaths"], ascending=False).head(n).reset_index(drop=True)  # 1건짜리 칸은 우연일 수 있어 제외
    who = {"all": "", "child": "어린이(초등학생 이하) ", "elderly": "고령(60대 이상) "}.get(victim, "")
    if t.empty:
        return Result("보행자 사고 다발 지점", f"{who}보행자 사고가 같은 지점(약 400m)에서 2건 이상 난 곳이 없습니다.")
    t["장소"] = [f"{i + 1}. {r.dong} {r.road}".strip() for i, r in t.iterrows()]
    y0, y1 = int(d.year.min()), int(d.year.max())
    share = t.n.sum() / len(d) * 100
    top = t.iloc[0]
    s = (f"{y0}~{y1}년 {who}보행자 사고 {len(d):,}건을 약 400m 격자로 묶으면, 사고가 가장 많은 지점은 '{top.장소[3:]}'"
         f"({int(top.n)}건, 사망 {int(top.deaths)}명)입니다. 상위 {len(t)}곳이 전체의 {share:.0f}%를 차지합니다. "
         "※ 약 2년치 자료를 격자(약 400m) 단위로 묶은 결과라, 같은 장소가 둘로 나뉘거나 인접한 지점이 합쳐질 수 있습니다.")
    table = t.rename(columns={"n": "사고 건수", "deaths": "사망자", "serious": "중상자", "vio": "주된 법규위반"})[["장소", "사고 건수", "사망자", "중상자", "주된 법규위반"]]
    m = t[["lat", "lon", "장소"]].copy()
    m["값"], m["크기"] = t.n, t.n
    m["설명"] = [f"보행자 사고 {r.n}건 (사망 {r.deaths}명, 중상 {r.serious}명)\n주된 법규위반: {r.vio}" for r in t.itertuples()]
    return Result(f"{who}보행자 사고 다발 지점", s, table, [{"kind": "bar", "x": "장소", "y": "사고 건수", "horizontal": True}], m,
                  years=(y0, y1), map_legend="원 크기·숫자 = 보행자 사고 건수 (약 400m 격자)")


# ------------------------------------------------------------- 법규위반 단속
def enforcement_overview() -> Result:
    if not _has("enf_monthly"):
        return Result("법규위반 단속", "단속 데이터가 아직 없습니다. 수집을 먼저 실행하세요.")
    e = _q("select year, kind, sum(count) c from enf_monthly group by year, kind")
    piv = e.pivot(index="kind", columns="year", values="c").fillna(0)
    piv["합계"] = piv.sum(axis=1)
    piv = piv.sort_values("합계", ascending=False)
    tot = float(piv["합계"].sum())
    out = piv.reset_index().rename(columns={"kind": "위반 유형"})
    out["비중(%)"] = (out["합계"] / tot * 100).round(1)
    out.columns = [str(c) for c in out.columns]
    yt = e.groupby("year").c.sum()
    y0, y1 = int(yt.index.min()), int(yt.index.max())
    med = float(yt.median())
    top = out.iloc[0]
    s = (f"{y0}~{y1}년 법규위반 단속은 모두 {tot:,.0f}건이고, 가장 많은 유형은 '{top['위반 유형']}'({top['비중(%)']}%)입니다. "
         f"연간 단속 건수는 {y0}년 {yt.iloc[0]:,.0f}건, {y1}년 {yt.iloc[-1]:,.0f}건입니다.")
    spikes = [(int(y), v / med) for y, v in yt.items() if len(yt) >= 4 and v / med >= 1.6]
    if spikes:
        s += " " + ", ".join(f"{y}년은 다른 해의 약 {r:.1f}배" for y, r in spikes) + "로 두드러져 단속 방침 변화 등 원인 확인이 필요합니다."
    s += " ※ 단속 건수는 위반이 실제로 일어난 건수가 아니라 단속 활동의 결과이므로, 단속이 늘었다고 위반이 늘었다는 뜻은 아닙니다."
    return Result("법규위반별 교통단속", s, out, [{"kind": "bar", "x": "위반 유형", "y": "합계", "horizontal": True}], years=(y0, y1))


def year_span() -> tuple[int, int]:
    """보유한 모든 표(연도 컬럼이 있는 것)의 가장 이른 연도와 가장 늦은 연도."""
    con = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
    try:
        lo, hi = [], []
        for (t,) in con.execute("select table_name from source_files").fetchall():
            if "year" in [r[1] for r in con.execute(f"pragma table_info({t})").fetchall()]:
                a, b = con.execute(f"select min(year), max(year) from {t}").fetchone()
                lo.append(a)
                hi.append(b)
        return int(min(lo)), int(max(hi))
    finally:
        con.close()


def year_notice(year=None) -> Result:
    lo, hi = year_span()
    y = f"{int(year)}년 " if year else ""
    return Result("자료 범위 밖", f"{y}자료는 보유하고 있지 않습니다. 보유한 자료는 표마다 기간이 다르며 {lo}~{hi}년 범위입니다. "
                  "앞으로의 건수는 '전망' 기능(예: 앞으로 1년 전체 사고를 전망해줘)으로 참고용 전망을 볼 수 있어요. 자세한 기간은 '데이터 현황' 메뉴에서 확인하세요.")


def improvement_advice(audience: str = "all") -> Result:
    """분석 결과를 근거로 한 개선 제안(도민 행동 안내 + 담당 기관 검토 제안)."""
    from . import advice
    items = advice.build()
    if audience in ("citizen", "admin"):
        items = [a for a in items if a.audience == audience]
    if not items:
        return Result("개선 제안", "제안을 만들 분석 결과가 아직 없습니다. 수집을 먼저 실행하세요.")
    adm, cit = [a for a in items if a.audience == "admin"], [a for a in items if a.audience == "citizen"]
    parts = []
    if adm:
        parts.append("담당 기관에는 다음을 검토해 볼 수 있습니다. " + " ".join(f"({i}) {a.title}: {a.basis}" for i, a in enumerate(adm[:3], 1)))
    if cit:
        parts.append("도민께는 다음을 권합니다. " + " ".join(f"({i}) {a.title}." for i, a in enumerate(cit[:3], 1)))
    hot = next((a for a in adm if a.map_df is not None), None)
    return Result("개선 제안", " ".join(parts) + " " + advice.DISCLAIMER, advice.to_frame(items), [],
                  hot.map_df if hot else None, map_legend=hot.map_legend if hot else "")


# ------------------------------------------------------------- 레지스트리
_YEAR = {"type": "integer", "description": "연도(예: 2018). 생략하면 전체 기간"}
TOOLS = {
    "data_overview": (data_overview, "보유 데이터의 종류·기간·행 수. 어떤 데이터가 있는지 묻거나 기간이 불확실할 때.", {}),
    "quality_report": (quality_report, "데이터의 문제점(규모 이상, 결측, 합계 불일치, 연도 정보 없음 등) 점검 결과.", {}),
    "risk_by_factor": (risk_by_factor, "요인별 사고 건수·치명도 비교. 교차로 형태, 도로선형(커브·오르막·내리막), 기상(비·눈·안개), 노면(젖음·결빙).", {
        "factor": {"type": "string", "enum": list(_FACTORS), "description": "intersection=교차로, alignment=도로선형, weather=기상, surface=노면"},
        "start_year": _YEAR, "end_year": _YEAR}),
    "monthly_trend": (monthly_trend, "월별 사고 추이·계절성. 노인 교통사고(elderly) 또는 전체 사고(intersection).", {
        "target": {"type": "string", "enum": list(_TREND)}, "start_year": _YEAR, "end_year": _YEAR}),
    "pedestrian_risk": (pedestrian_risk, "보행자 사고를 시간대·요일·기상·노면·법규위반·피해자 연령·도로형태·지역별로 비교(사고 단위 자료).", {
        "ped_by": {"type": "string", "enum": list(_PED_BY)}}),
    "pedestrian_cross": (pedestrian_cross, "보행자 사고를 두 요인으로 교차 분석(예: 기상×시간대, 연령×시간대, 법규위반×연령).", {
        "rows": {"type": "string", "enum": list(_PED_BY)}, "cols": {"type": "string", "enum": list(_PED_BY)}}),
    "pedestrian_hotspots": (pedestrian_hotspots, "보행자 사고가 실제로 많이 난 지점(약 400m 격자)과 지도. 어린이·고령 피해자만 볼 수도 있음.", {
        "city": {"type": "string", "description": "제주시 또는 서귀포시"}, "n": {"type": "integer", "description": "상위 몇 곳"},
        "victim": {"type": "string", "enum": ["all", "child", "elderly"]}}),
    "improvement_advice": (improvement_advice, "분석 결과를 근거로 한 개선 제안. 사고를 줄이려면·대책·조심할 점·개선이 필요한 곳을 물을 때. audience: citizen(도민 행동 안내), admin(담당 기관 제안), all.", {"audience": {"type": "string", "enum": ["all", "citizen", "admin"]}}),
    "year_notice": (year_notice, "보유 기간 밖의 연도(예: 먼 미래, 아주 오래전)를 물을 때 자료가 없다고 안내.", {"year": {"type": "integer"}}),
    "enforcement_overview": (enforcement_overview, "법규위반별 교통단속(과속·신호위반·음주 등) 건수와 연도별 추이.", {}),
    "forecast": (forecast, "향후 월별 건수 전망(예측 범위와 검증 오차 포함). 노인 사고(elderly), 전체 사고(intersection), 단속 건수(enforcement). 앞으로·전망·예측 질문.", {
        "target": {"type": "string", "enum": list(_FC_TARGETS)}, "horizon": {"type": "integer", "description": "전망할 개월 수(최대 12)"}}),
    "child_risk": (child_risk, "12세 이하 어린이 교통사고를 시간대(hour) 또는 도로 종류(road)별로.", {
        "by": {"type": "string", "enum": ["hour", "road"]}, "start_year": _YEAR, "end_year": _YEAR}),
    "hotspot_rank": (hotspot_rank, "사고 다발지역 순위와 지도. 도시(제주시/서귀포시) 지정 가능.", {
        "city": {"type": "string", "description": "제주시 또는 서귀포시"}, "n": {"type": "integer", "description": "상위 몇 곳"}}),
}


def tool_schemas() -> list[dict]:
    return [{"type": "function", "function": {"name": k, "description": d,
            "parameters": {"type": "object", "properties": p, "required": []}}} for k, (_, d, p) in TOOLS.items()]


def run_tool(name: str, args: dict | None) -> Result:
    if name not in TOOLS:
        return Result("오류", f"알 수 없는 도구입니다: {name}")
    fn, _, props = TOOLS[name]
    clean = {k: v for k, v in (args or {}).items() if k in props and v not in (None, "")}
    try:
        return fn(**clean)
    except FileNotFoundError as e:
        return Result("데이터 없음", str(e))
    except Exception as e:  # LLM 이 잘못된 인자를 줘도 앱이 죽지 않게
        return Result("분석 오류", f"분석 중 문제가 발생했습니다({type(e).__name__}). 질문이나 기간을 바꿔 보세요.")
