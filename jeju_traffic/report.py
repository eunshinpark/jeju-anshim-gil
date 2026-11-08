"""자동 점검 보고서.

버튼 한 번으로 전체 분석(데이터 신뢰도 → 핵심 발견 → 요인별 → 취약계층 → 다발지역 → 한계)을 돌려
HTML(인터넷 없이 열림, 인쇄/PDF 저장 가능)과 Markdown 으로 출력한다.
숫자는 모두 분석 도구가 계산한 값이고, AI 요약은 선택 사항이며 근거 없는 숫자가 있으면 폐기한다.

    python -m jeju_traffic.report            # reports/ 폴더에 .html, .md 생성
"""
from __future__ import annotations

import html
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import analysis as A
from . import advice, diagrams, maps
from .config import HOTSPOT_PERIOD, ROOT, db_path


@dataclass
class Finding:
    headline: str
    evidence: str
    caution: str = ""
    section: str = ""


@dataclass
class Block:
    kind: str  # table | bars | line
    title: str = ""
    df: pd.DataFrame | None = None
    x: str = ""
    y: str = ""
    html: str = ""  # kind == 'html' 일 때 그대로 넣는 HTML(도로선형 그림 등)
    legend: str = ""  # kind == 'map' 일 때 지도 범례
    only: str = ""  # "md"=Markdown 에서만, ""=화면·HTML 모두(Markdown 은 표만 사용)


@dataclass
class Section:
    title: str
    paragraphs: list[str] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Report:
    title: str
    generated_at: str
    findings: list[Finding]
    sections: list[Section]
    ai_summary: str | None = None


# ---------------------------------------------------------------- 보조
def _row(df: pd.DataFrame, label: str):
    r = df[df.구분 == label]
    return r.iloc[0] if len(r) else None


def _yr(res) -> str:
    """결과가 사용한 연도 범위를 문구로 (데이터에서 읽은 값)."""
    a, b = res.years or (None, None)
    return "" if a is None else (f"{a}년" if a == b else f"{a}~{b}년")


def _ratio(a, b):
    return a / b if b and b > 0 else None


def _fmt_n(x) -> str:
    return f"{int(x):,}"


# ---------------------------------------------------------------- 핵심 발견 (결과표에서 규칙으로 추출)
def _findings(R: dict[str, A.Result]) -> list[Finding]:
    out: list[Finding] = []
    w = R["weather"].df
    r, b = _row(w, "비"), _row(w, "맑음")
    if r is not None and b is not None and (x := _ratio(r.치명도, b.치명도)):
        out.append(Finding(
            f"비 오는 날은 맑은 날보다 사고 100건당 사망자가 {x:.1f}배 많았습니다.",
            f"비 {r.치명도}명(사고 {_fmt_n(r['사고 건수'])}건) vs 맑음 {b.치명도}명(사고 {_fmt_n(b['사고 건수'])}건), {_yr(R['weather'])}.",
            "교통량을 반영하지 않은 비교이며, 비 오는 날은 사고 건수 자체도 적어 값이 흔들릴 수 있습니다.", "요인별"))
    s = R["surface"].df
    wet, dry = _row(s, "젖음/습기"), _row(s, "건조")
    if wet is not None and dry is not None and (x := _ratio(wet.치명도, dry.치명도)):
        out.append(Finding(
            f"노면이 젖은 날은 건조한 날보다 치명도가 {x:.1f}배 높았습니다.",
            f"젖음/습기 {wet.치명도}명(사고 {_fmt_n(wet['사고 건수'])}건) vs 건조 {dry.치명도}명(사고 {_fmt_n(dry['사고 건수'])}건).",
            "결빙·적설은 사고 수가 적어 비교가 불안정합니다.", "요인별"))
    al = R["alignment"].df
    base = _row(al, "직선 평지")
    big = al[(al["사고 건수"] >= 30) & (al.구분 != "직선 평지")]
    if base is not None and len(big):
        top = big.sort_values("치명도", ascending=False).iloc[0]
        if x := _ratio(top.치명도, base.치명도):
            out.append(Finding(
                f"'{top.구분}' 구간의 치명도는 직선 평지의 {x:.1f}배였습니다.",
                f"{top.치명도}명(사고 {_fmt_n(top['사고 건수'])}건) vs 직선 평지 {base.치명도}명(사고 {_fmt_n(base['사고 건수'])}건), {_yr(R['alignment'])}.",
                "커브·경사 구간의 사고 수가 적어 연도별 변동이 큽니다.", "요인별"))
    ix = R["intersection"].df
    no = _row(ix, "교차로 아님")
    if no is not None:
        inter = ix[ix.구분 != "교차로 아님"]
        ia = inter["사고 건수"].sum()
        out.append(Finding(
            f"사고의 {100 - no['구성비(%)']:.1f}%는 교차로에서, {no['구성비(%)']}%는 교차로 밖에서 발생했습니다.",
            f"교차로 {_fmt_n(ia)}건 / 교차로 아님 {_fmt_n(no['사고 건수'])}건, {_yr(R['intersection'])}. 교차로 안에서는 '{inter.sort_values('사고 건수', ascending=False).iloc[0].구분}' 형태가 가장 많았습니다.",
            "교차로 개수·통행량 대비 위험도는 알 수 없습니다.", "요인별"))
    ch = R["child_hour"].df
    t3 = ch.sort_values("사고 건수", ascending=False).head(3)
    out.append(Finding(
        f"어린이(12세 이하) 사고는 {', '.join(t3.구분)}에 집중되었습니다(합계 {t3['구성비(%)'].sum():.0f}%).",
        f"가장 많은 시간대는 {t3.iloc[0].구분}({_fmt_n(t3.iloc[0]['사고 건수'])}건, {t3.iloc[0]['구성비(%)']}%), {_yr(R['child_hour'])}.",
        "같은 시간대 어린이 통행량은 반영되지 않았습니다.", "취약계층"))
    cr = R["child_road"].df
    out.append(Finding(
        f"어린이 사고의 {cr.iloc[0]['구성비(%)']}%가 '{cr.iloc[0].구분}'에서 발생했습니다.",
        f"{_fmt_n(cr.iloc[0]['사고 건수'])}건, {_yr(R['child_road'])}.", "도로 종류별 연장·통행량 차이는 반영되지 않았습니다.", "취약계층"))
    el = R["elderly"].df
    el = el.assign(연=el.월.str[:4])
    yr = el.groupby("연")["사고 건수"].sum()
    chg = (yr.iloc[-1] / yr.iloc[0] - 1) * 100
    out.append(Finding(
        f"노인 교통사고는 {yr.index[0]}년 {_fmt_n(yr.iloc[0])}건에서 {yr.index[-1]}년 {_fmt_n(yr.iloc[-1])}건으로 {abs(chg):.0f}% {'증가' if chg > 0 else '감소'}했습니다.",
        f"사망자는 {yr.index[0]}년 {_fmt_n(el[el.연 == yr.index[0]]['사망자'].sum())}명, {yr.index[-1]}년 {_fmt_n(el[el.연 == yr.index[-1]]['사망자'].sum())}명.",
        "관찰 기간의 추이이며 원인을 단정할 수는 없습니다.", "취약계층"))
    h = R["hotspot"].df
    for city in ("제주시", "서귀포시"):
        c = h[h.도시 == city]
        if len(c):
            t = c.iloc[0]
            out.append(Finding(
                f"{city}에서 순위표에 가장 자주 오른 곳은 '{t.장소}'입니다({int(t.등장횟수)}개 시점).",
                f"시점당 평균 사고 {t.평균사고}건, 최대 {int(t.최대사고)}건.",
                f"시점별 연도({HOTSPOT_PERIOD} 중)를 확인할 수 없어 연도별 비교는 하지 않았습니다.", "다발지역"))
    ph, pa, pv, pt, en = R["ped_hour"], R["ped_age"], R["ped_vio"], R["ped_hot"], R["enf"]
    if len(ph.df):
        t3 = ph.df.sort_values("사고 건수", ascending=False).head(3)
        out.append(Finding(
            f"보행자 사고의 {t3['구성비(%)'].sum():.0f}%가 {', '.join(t3.구분)}에 몰렸습니다.",
            f"전체 {int(ph.df['사고 건수'].sum()):,}건 중 가장 많은 시간대는 {t3.iloc[0].구분}({int(t3.iloc[0]['사고 건수'])}건), {_yr(ph)}.",
            "보행량이 많은 시간대일 수 있어 건수만으로 위험도를 말할 수 없습니다.", "보행자"))
    if len(pa.df) and pa.df["사망자"].sum() > 0:
        old = pa.df[pa.df.구분.isin(["60대", "70대", "80대", "90대이상"])]
        out.append(Finding(
            f"보행자 사고 피해자의 {old['구성비(%)'].sum():.0f}%가 60대 이상이고, 사망자의 {old['사망자'].sum() / pa.df['사망자'].sum() * 100:.0f}%를 차지합니다.",
            f"60대 이상 {int(old['사고 건수'].sum()):,}건, 사망 {int(old['사망자'].sum())}명 (전체 사망 {int(pa.df['사망자'].sum())}명), {_yr(pa)}.",
            "연령대별 보행 인구는 반영되지 않았습니다.", "보행자"))
    if len(pv.df):
        v = pv.df.iloc[0]
        out.append(Finding(
            f"보행자 사고에서 가해 운전자의 법규위반은 '{v.구분}'이 가장 많았습니다({v['구성비(%)']}%).",
            f"{int(v['사고 건수']):,}건, {_yr(pv)}.", "법규위반 유형은 사고 당시의 경찰 기록 기준이며 과실 비율을 뜻하지 않습니다.", "보행자"))
    if len(pt.df):
        h1 = pt.df.iloc[0]
        out.append(Finding(
            f"보행자 사고가 가장 많이 난 지점은 '{str(h1.장소)[3:]}'입니다({int(h1['사고 건수'])}건).",
            f"사망 {int(h1['사망자'])}명, 중상 {int(h1['중상자'])}명, {_yr(pt)}. 약 400m 격자 기준.", "격자 경계에서 같은 장소가 나뉠 수 있습니다.", "보행자"))
    if len(en.df):
        e1 = en.df.iloc[0]
        out.append(Finding(
            f"법규위반 단속의 {e1['비중(%)']}%는 '{e1['위반 유형']}'입니다.",
            f"{_yr(en)} 합계 {int(en.df['합계'].sum()):,}건 중 {int(e1['합계']):,}건.",
            "단속 건수는 단속 활동의 결과이며 위반 발생 건수가 아닙니다.", "단속"))
    for k, name, extra in (("fc_total", "전체 교통사고", ""), ("fc_elderly", "노인 교통사고", " 노인 사고는 월 건수가 적어 오차가 큽니다."),
                           ("fc_enf", "법규위반 단속 건수", "")):
        m = R[k].meta
        if m:
            out.append(Finding(
                f"향후 {m['horizon']}개월 {name}는 {m['peak']}월에 가장 많을 것(약 {m['peak_v']:,.0f}건)으로 전망됩니다.",
                f"{m['m0']} ~ {m['m1']}, 최근 12개월 수준이 이어진다는 가정(합계 약 {m['total']:,.0f}건). "
                f"검증 평균 오차 {m['mape'] * 100:.0f}% (단순 방법 {m['mape_naive'] * 100:.0f}%).",
                "참고용 전망입니다. 증가·감소 추세와 날씨·교통량·정책 변화는 반영하지 않았습니다." + extra, "전망"))
    return out


# ---------------------------------------------------------------- 신뢰도 섹션
ACTIONS = {
    "SCALE_MISMATCH": "규모 이상으로 표시된 연도를 분석에서 제외했습니다.",
    "REPEATED_RANKS": "여러 시점의 순위표를 합친 자료로 취급하고, 연도별 비교 대신 '순위표 등장 횟수'로 분석했습니다.",
    "NO_REFERENCE_YEAR": "다발지역의 연도를 단정하지 않았습니다.",
    "MISSING_VALUES": "빈 값은 합산에서 제외했습니다(0건으로 간주하지 않음).",
    "IMPUTED_ZERO": "사고가 없어 행이 빠진 시간대를 0건으로 채웠습니다.",
    "TOTAL_MISMATCH": "합계 행과 세부 합이 달라 확인이 필요합니다.",
    "GEO_OUT": "제주 범위를 벗어난 좌표는 지도·지점 분석에서 제외했습니다.",
    "SPIKE_YEAR": "해당 연도를 그대로 표시하되, 추세·전망 해석에 주의하도록 안내했습니다.",
    "MONTH_GAP": "빠진 달이 있어 해당 자료의 월별 전망은 하지 않습니다.",
}


# ---------------------------------------------------------------- 보고서 조립
def build(use_llm: bool = False) -> Report:
    _w = A.risk_by_factor("weather")
    R = {
        "weather": _w, "surface": A.risk_by_factor("surface", _w.years[0] if _w.years else None),
        "alignment": A.risk_by_factor("alignment"), "intersection": A.risk_by_factor("intersection"),
        "child_hour": A.child_risk("hour"), "child_road": A.child_risk("road"),
        "elderly": A.monthly_trend("elderly"), "total": A.monthly_trend("intersection"),
        "hotspot": A.hotspot_rank(None, 12),
        "fc_total": A.forecast("intersection"), "fc_elderly": A.forecast("elderly"), "fc_enf": A.forecast("enforcement"),
        "ped_hour": A.pedestrian_risk("hour"), "ped_age": A.pedestrian_risk("victim_age"), "ped_vio": A.pedestrian_risk("violation"),
        "ped_cross": A.pedestrian_cross("victim_age", "hour"), "ped_hot": A.pedestrian_hotspots(None, 10), "enf": A.enforcement_overview(),
    }
    findings = _findings(R)

    secs = []  # 데이터 현황·신뢰도는 앱의 '데이터 현황' 탭에서 본다
    f = Section("1. 위험 요인별 분석")
    f.paragraphs.append("기상·노면·도로선형·교차로 형태별 사고 건수와 치명도(사고 100건당 사망자)입니다. " + A.CAVEAT)
    for k, name in (("weather", "기상 상태별"), ("surface", "노면 상태별"), ("alignment", "도로선형별"), ("intersection", "교차로 형태별")):
        f.paragraphs.append(R[k].summary.split(" ※")[0])
        if k == "alignment":
            f.blocks.append(Block("html", "도로선형 그림", html=diagrams.alignment_cards(R[k].df)))
        if k == "intersection":
            f.blocks.append(Block("html", "교차로 형태 그림", html=diagrams.intersection_cards(R[k].df)))
        f.blocks.append(Block("bars", f"{name} 사고 건수", R[k].df, "구분", "사고 건수"))
        f.blocks.append(Block("table", f"{name} 상세", R[k].df))
    secs.append(f)

    v = Section("2. 어린이·노인 교통사고")
    v.paragraphs += [R["child_hour"].summary.split(" ※")[0], R["child_road"].summary.split(" ※")[0],
                     R["elderly"].summary.split(" ※")[0]]
    v.blocks += [Block("bars", "어린이 사고 - 시간대별", R["child_hour"].df, "구분", "사고 건수"),
                 Block("bars", "어린이 사고 - 도로 종류별", R["child_road"].df, "구분", "사고 건수"),
                 Block("line", "노인 교통사고 월별 추이", R["elderly"].df, "월", "사고 건수"),
                 Block("line", "전체 교통사고 월별 추이", R["total"].df, "월", "사고 건수")]
    secs.append(v)

    pd_ = Section("3. 보행자 사고·법규위반 단속")
    for key in ("ped_hour", "ped_age", "ped_cross", "ped_hot", "enf"):
        if len(R[key].df):
            pd_.paragraphs.append(R[key].summary.split(" ※")[0])
    if len(R["ped_hour"].df):
        pd_.blocks.append(Block("bars", "보행자 사고 - 시간대별", R["ped_hour"].df, "구분", "사고 건수"))
    if len(R["ped_age"].df):
        pd_.blocks.append(Block("bars", "보행자 사고 - 피해자 연령대별", R["ped_age"].df, "구분", "사고 건수"))
    if len(R["ped_cross"].df):
        pd_.blocks.append(Block("table", "보행자 사고 - 연령대 × 시간대", R["ped_cross"].df))
    if len(R["ped_hot"].df):
        pd_.blocks.append(Block("map", "보행자 사고 다발 지점 지도", R["ped_hot"].map_df, legend=R["ped_hot"].map_legend))
        pd_.blocks.append(Block("table", "보행자 사고 다발 지점(약 400m 격자)", R["ped_hot"].df))
    if len(R["enf"].df):
        pd_.blocks.append(Block("bars", "법규위반별 단속 건수", R["enf"].df, "위반 유형", "합계"))
    pd_.paragraphs.append(A.PED_CAVEAT)
    secs.append(pd_)

    h = Section("4. 사고 다발지역")
    h.paragraphs.append(R["hotspot"].summary)
    if R["hotspot"].map_df is not None:
        h.blocks.append(Block("map", "사고 다발지역 지도", R["hotspot"].map_df, legend=R["hotspot"].map_legend))
    h.blocks.append(Block("table", "순위표에 자주 오른 곳", R["hotspot"].df))
    secs.append(h)

    fc = Section("5. 향후 전망 (참고)")
    for k, name in (("fc_total", "전체 교통사고"), ("fc_elderly", "노인 교통사고"), ("fc_enf", "법규위반 단속 건수")):
        res = R[k]
        fc.paragraphs.append(res.summary.split(" ※")[0])
        if len(res.df):
            val = res.meta.get("val", "사고 건수")
            fc.blocks.append(Block("forecast", f"{name} 월별 전망", res.df, "월", val))
            fut = res.df[res.df.구분 == "전망"][["월", val, "하한", "상한"]].rename(
                columns={val: "전망(건)", "하한": "80% 하한", "상한": "80% 상한"})
            fc.blocks.append(Block("table", f"{name} 전망 값", fut))
    fc.paragraphs.append(A.FC_CAVEAT)
    secs.append(fc)

    ad = Section("6. 개선 제안")
    try:
        advs = advice.build()
    except Exception:
        advs = []
    if advs:
        ad.paragraphs.append("분석 결과를 근거로 한 제안입니다. 각 제안에는 근거 수치와 한계, 근거의 강도(표본 크기 기준)를 함께 적었습니다.")
        for who, name in (("admin", "담당 기관이 검토해 볼 제안"), ("citizen", "도민 행동 안내")):
            sub = [a for a in advs if a.audience == who]
            if sub:
                ad.blocks.append(Block("html", name, html=diagrams.advice_cards(sub)))  # 화면·HTML: 일러스트 카드
                ad.blocks.append(Block("table", name, advice.to_frame(sub).drop(columns=["대상"]), only="md"))  # Markdown: 표
        ad.paragraphs.append(advice.DISCLAIMER)
        secs.append(ad)

    lim = Section("7. 한계와 해석 시 주의사항")
    lim.paragraphs += [
        "사고 건수는 도로·시간대별 교통량(노출도)을 반영하지 않았습니다. '건수가 많다'는 것이 '더 위험하다'는 뜻은 아닙니다.",
        "치명도는 사고 100건당 사망자 수이며, 사고가 적은 구분(눈, 결빙, 커브 일부 등)은 값이 크게 흔들립니다.",
        "분석 기간 밖의 변화(교통량·도로 환경의 변화 등)는 반영되지 않았습니다. 기간은 각 분석 결과에 표시했습니다.",
        "기상·노면·도로선형·교차로 파일은 각각 따로 집계된 표라서 '비 오는 날의 커브길'처럼 요인을 겹쳐 분석할 수 없습니다.",
        "분석 결과는 요인 간 관계(상관)를 보여줄 뿐 원인을 증명하지 않습니다.",
    ]
    secs.append(lim)

    ap = Section("부록. 분석 방법")
    ap.paragraphs += [
        "치명도 = 사망자 ÷ 사고 건수 × 100. 건당 부상 = 부상자 ÷ 사고 건수.",
        "합계·소계 행은 세부 행과 이중 집계되지 않도록 구분(level)해 세부 행만 합산했습니다.",
        "기상 파일은 규모 점검을 통과한 연도만 사용했습니다(같은 해 노면 파일과 건수 규모를 비교해 판정).",
        "모든 수치는 코드가 계산했으며, AI(언어모델)는 요약 문장을 쓰는 데에만 선택적으로 사용되고 근거 없는 숫자가 있으면 폐기됩니다.",
        "사용한 데이터의 종류·기간과 자동 품질 점검 결과(발견한 문제와 반영한 조치)는 앱의 '데이터 현황' 메뉴에서 볼 수 있습니다.",
    ]
    secs.append(ap)

    rep = Report("제주 안심길 · 교통사고 자동 점검 보고서", datetime.now().strftime("%Y-%m-%d %H:%M"), findings, secs)
    if use_llm:
        rep.ai_summary = _ai_summary(findings)
    return rep


def _ai_summary(findings: list[Finding]) -> str | None:
    """핵심 발견을 3~4문장으로 요약. 근거 없는 숫자가 있으면 None."""
    from . import agent
    try:
        if not agent.llm_available():
            return None
        ev = "\n".join(f"- {f.headline} ({f.evidence})" for f in findings)
        txt = agent._chat([
            {"role": "system", "content": "교통안전 보고서 작성자입니다. 아래 핵심 발견만 근거로 3~4문장으로 요약하세요. 새로운 숫자를 만들지 말고, 원인을 단정하지 말고 '경향'으로 쓰세요."},
            {"role": "user", "content": ev}]).strip()
        return txt if txt and agent._numbers_grounded(txt, ev) else None
    except Exception:
        return None


# ---------------------------------------------------------------- 렌더링
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """값이 모두 정수인 실수 컬럼(결측 때문에 float 가 된 경우)을 정수로 되돌린다."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]) and d[c].dropna().map(float.is_integer).all():
            d[c] = d[c].astype("Int64")
    return d


def _md_table(df: pd.DataFrame) -> str:
    df = _clean(df)
    d = df.copy().astype(object).where(df.notna(), "")
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    rows = "".join("| " + " | ".join(str(v).replace("|", "/") for v in r) + " |\n" for r in d.itertuples(index=False))
    return head + rows


def to_markdown(rep: Report) -> str:
    o = [f"# {rep.title}\n", f"작성: {rep.generated_at} · 제주데이터허브 교통사고 데이터 자동 분석\n"]
    if rep.ai_summary:
        o.append(f"## AI 요약\n{rep.ai_summary}\n")
    o.append("## 핵심 발견\n")
    for i, f in enumerate(rep.findings, 1):
        o.append(f"{i}. **{f.headline}**  \n   근거: {f.evidence}  \n   ⚠ {f.caution}\n")
    for s in rep.sections:
        o.append(f"\n## {s.title}\n")
        o += [p + "\n" for p in s.paragraphs]
        for b in s.blocks:
            if b.kind == "table" and b.df is not None:
                o.append(f"\n**{b.title}**\n\n{_md_table(b.df)}")
    return "\n".join(o)


_CSS = """
:root{--fg:#333;--mut:#666;--line:#e1e1e1;--acc:#224da0;--bg:#f4f6fb;--warn:#9a6700}
*{box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,'Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo',Helvetica,Arial,sans-serif;color:var(--fg);max-width:960px;margin:0 auto;padding:24px 16px;line-height:1.65}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.meta{color:var(--mut);font-size:13px}.box{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:14px 16px;margin:12px 0}
.find{margin:10px 0;padding:12px 14px;border-left:4px solid var(--acc);background:#fff;border-radius:6px}
.find b{display:block;margin-bottom:2px}.ev{font-size:14px;color:var(--mut)}.cau{font-size:13px;color:var(--warn)}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 18px}th,td{border:1px solid var(--line);padding:5px 8px;text-align:left}
th{background:var(--bg)}h3{font-size:15px;margin:18px 0 4px}
.bar{display:flex;align-items:center;margin:3px 0;font-size:13px}.bar .l{width:170px;flex:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar .t{flex:1;background:var(--bg);border-radius:4px;height:16px}.bar .f{height:16px;background:var(--acc);border-radius:4px}.bar .v{width:70px;text-align:right;flex:none}
svg{width:100%;height:auto;background:var(--bg);border:1px solid var(--line);border-radius:8px}
@media print{body{max-width:none}.find{break-inside:avoid}h2{break-after:avoid}}
"""


def _bars(df, x, y):
    m = float(df[y].max()) or 1.0
    rows = "".join(
        f"<div class='bar'><span class='l' title='{html.escape(str(r[x]))}'>{html.escape(str(r[x]))}</span>"
        f"<span class='t'><span class='f' style='display:block;width:{max(float(r[y]) / m * 100, 0.5):.1f}%'></span></span>"
        f"<span class='v'>{float(r[y]):,.0f}</span></div>" for _, r in df.iterrows())
    return rows


def _line(df, x, y):
    ys = df[y].astype(float).tolist()
    W, H, P = 640, 200, 30
    lo, hi = min(ys), max(ys)
    span = (hi - lo) or 1
    pts = [(P + i * (W - 2 * P) / max(len(ys) - 1, 1), H - P - (v - lo) / span * (H - 2 * P)) for i, v in enumerate(ys)]
    poly = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    ticks = "".join(f"<text x='{pts[i][0]:.0f}' y='{H - 8}' font-size='10' text-anchor='middle' fill='#666666'>{html.escape(str(df[x].iloc[i]))}</text>"
                    for i in range(0, len(ys), max(len(ys) // 6, 1)))
    return (f"<svg viewBox='0 0 {W} {H}'><polyline points='{poly}' fill='none' stroke='#2f6fed' stroke-width='2'/>"
            f"<text x='{P}' y='14' font-size='11' fill='#666666'>최대 {hi:,.0f} / 최소 {lo:,.0f}</text>{ticks}</svg>")


def _forecast_svg(df, x, y):
    a, f = df[df.구분 == "실적"], df[df.구분 == "전망"]
    vals = list(a[y]) + list(f[y]) + list(f["상한"]) + list(f["하한"])
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1
    n, k = len(a) + len(f), len(a)
    W, H, P = 640, 230, 30
    px = lambda i: P + i * (W - 2 * P) / max(n - 1, 1)
    py = lambda v: H - P - (v - lo) / span * (H - 2 * P)
    act = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(a[y]))
    fc = " ".join(f"{px(k - 1 + i):.1f},{py(v):.1f}" for i, v in enumerate([a[y].iloc[-1]] + list(f[y])))
    up = [f"{px(k + i):.1f},{py(v):.1f}" for i, v in enumerate(f["상한"])]
    dn = [f"{px(k + i):.1f},{py(v):.1f}" for i, v in enumerate(f["하한"])][::-1]
    months = list(a[x]) + list(f[x])
    ticks = "".join(f"<text x='{px(i):.0f}' y='{H - 8}' font-size='10' text-anchor='middle' fill='#666666'>{html.escape(months[i])}</text>"
                    for i in range(0, n, max(n // 7, 1)))
    return (f"<svg viewBox='0 0 {W} {H}'><polygon points='{' '.join(up + dn)}' fill='#224da0' fill-opacity='.16'/>"
            f"<polyline points='{act}' fill='none' stroke='#224da0' stroke-width='2'/>"
            f"<polyline points='{fc}' fill='none' stroke='#224da0' stroke-width='2' stroke-dasharray='4 4'/>"
            f"<text x='{P}' y='14' font-size='11' fill='#666666'>실선=실적, 점선=전망, 음영=80% 범위</text>{ticks}</svg>")


def _map_iframe(df, legend) -> str:
    """보고서 HTML 에 지도를 넣는다(OpenStreetMap, 키 불필요). 인터넷이 없으면 안내 문구가 나온다."""
    doc = maps.leaflet_map_html(df, legend, 420)
    return f"<iframe style='width:100%;height:430px;border:0' loading='lazy' srcdoc=\"{html.escape(doc, quote=True)}\"></iframe>"


def _html_table(df: pd.DataFrame) -> str:
    df = _clean(df)
    d = df.astype(object).where(df.notna(), "")
    th = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    tr = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in d.itertuples(index=False))
    return f"<table><tr>{th}</tr>{tr}</table>"


def to_html(rep: Report) -> str:
    e = html.escape
    o = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>{e(rep.title)}</title><style>{_CSS}</style></head><body>",
         f"<h1>{e(rep.title)}</h1><div class='meta'>작성: {e(rep.generated_at)} · 제주데이터허브 교통사고 데이터 자동 분석</div>"]
    if rep.ai_summary:
        o.append(f"<div class='box'><b>AI 요약</b><br>{e(rep.ai_summary)}</div>")
    o.append("<h2>핵심 발견</h2>")
    for i, f in enumerate(rep.findings, 1):
        o.append(f"<div class='find'><b>{i}. {e(f.headline)}</b><div class='ev'>근거: {e(f.evidence)}</div>"
                 f"<div class='cau'>⚠ {e(f.caution)}</div></div>")
    for s in rep.sections:
        o.append(f"<h2>{e(s.title)}</h2>")
        o += [f"<p>{e(p)}</p>" for p in s.paragraphs]
        for b in s.blocks:
            if b.only == "md":
                continue
            o.append(f"<h3>{e(b.title)}</h3>")
            if b.kind == "html":
                o.append(b.html)
                continue
            if b.df is None:
                continue
            o.append(_bars(b.df, b.x, b.y) if b.kind == "bars" else _line(b.df, b.x, b.y) if b.kind == "line"
                     else _forecast_svg(b.df, b.x, b.y) if b.kind == "forecast"
                     else _map_iframe(b.df, b.legend) if b.kind == "map" else _html_table(b.df))
    o.append("</body></html>")
    return "".join(o)


LATEST = ROOT / "reports" / "latest.pkl"
REPORT_VERSION = 5  # 보고서 구성(섹션·블록)이 바뀌면 올린다. 다르면 저장본을 쓰지 않고 새로 만들게 한다.


def save_latest(rep: Report) -> None:
    """앱을 다시 열어도 바로 보여 줄 수 있도록 마지막으로 만든 보고서를 저장한다."""
    try:
        LATEST.parent.mkdir(parents=True, exist_ok=True)
        LATEST.write_bytes(pickle.dumps({"v": REPORT_VERSION, "rep": rep}))
    except Exception:
        pass  # 저장에 실패해도 보고서는 화면에 그대로 보인다


def load_latest() -> Report | None:
    try:
        data = pickle.loads(LATEST.read_bytes())
        rep = data.get("rep") if isinstance(data, dict) and data.get("v") == REPORT_VERSION else None
        return rep if isinstance(rep, Report) else None
    except Exception:  # 파일이 없거나, 코드가 바뀌어 읽을 수 없으면 새로 만들게 한다
        return None


def latest_is_stale() -> bool:
    """저장된 보고서보다 DB 가 더 나중에 갱신되었는지(데이터를 다시 수집했는지)."""
    try:
        return db_path().stat().st_mtime > LATEST.stat().st_mtime
    except OSError:
        return False


def save(rep: Report, out_dir: Path | None = None) -> tuple[Path, Path]:
    out = out_dir or ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    h, m = out / f"제주안심길_점검보고서_{stamp}.html", out / f"제주안심길_점검보고서_{stamp}.md"
    h.write_text(to_html(rep), encoding="utf-8")
    m.write_text(to_markdown(rep), encoding="utf-8")
    return h, m


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    rep = build(use_llm="--ai" in sys.argv)
    h, m = save(rep)
    save_latest(rep)
    print(f"핵심 발견 {len(rep.findings)}개, 섹션 {len(rep.sections)}개\n  {h}\n  {m}")
