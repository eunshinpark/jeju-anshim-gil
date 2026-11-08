"""개선 제안(대안): 분석 결과에서 '근거 → 제안 → 한계 → 신뢰'를 규칙으로 만든다.

원칙
  1. 근거 수치가 있는 제안만 만든다. 수치는 모두 분석 도구가 계산한 값이며 이 파일에는 숫자를 직접 적지 않는다.
  2. "~하라"가 아니라 "~를 점검·검토한다"로 쓴다(통계는 원인과 효과를 증명하지 않는다).
  3. 신뢰(근거 강함/보통/약함)는 표본 크기로 정한다: 사고 300건 이상=강함, 100건 이상=보통, 그 미만=약함.
     요인을 비교하는 제안(예: 치명도 배수)은 교통량이 없어 최대 '보통'으로 제한한다.
  4. 모든 제안에 한계를 붙이고, 일반 안전 수칙처럼 데이터로 직접 확인되지 않는 부분은 그렇다고 밝힌다.
대상은 두 가지다: 도민(행동 안내)과 담당 기관(점검·검토 제안).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import analysis as A

DISCLAIMER = ("※ 이 제안은 사고 통계에서 보이는 경향에 근거한 '검토 의견'입니다. 통계는 원인이나 개선 효과를 증명하지 않고, 교통량·보행량이 "
              "반영되지 않았으며, 이미 시행 중인 정책은 알 수 없습니다. 실제 적용은 현장 조사와 전문가 검토를 거쳐 판단해야 합니다.")
_RANK = {"강함": 0, "보통": 1, "약함": 2}
_OLD = ["60대", "70대", "80대", "90대이상"]


@dataclass
class Advice:
    audience: str  # "citizen"(도민) | "admin"(담당 기관)
    title: str
    basis: str     # 근거(분석 결과)
    action: str    # 제안(검토 내용)
    caveat: str    # 한계
    strength: str  # 강함 | 보통 | 약함
    source: str    # 근거가 된 분석
    items: list[str] = field(default_factory=list)  # 우선 점검 지점 등
    map_df: pd.DataFrame | None = None
    map_legend: str = ""
    art: str = ""  # 카드에 붙일 일러스트 종류(diagrams.advice_art 의 키)


# 제안 제목에 들어 있는 말 -> 일러스트. 위에서부터 먼저 맞는 것을 쓴다.
_ART_RULES = [("다발 지점", "hotspot"), ("집중 시간대", "peak_hours"), ("해질녘", "sunset_cross"), ("고령", "elderly"), ("어르신", "elderly"),
              ("단속", "enforcement"), ("구간 점검", "curve_down"), ("커브길", "curve_down"), ("교차로", "intersection"), ("우천", "rain"),
              ("비 오는 날", "rain"), ("통학", "child"), ("하교", "child"), ("횡단보도 앞", "crosswalk_slow")]


def _art_for(title: str) -> str:
    return next((k for word, k in _ART_RULES if word in title), "")


def _strength(n: int, comparative: bool) -> str:
    s = "강함" if n >= 300 else "보통" if n >= 100 else "약함"
    return "보통" if comparative and s == "강함" else s


def _get(fn, *args):
    try:
        r = fn(*args)
        return r if len(r.df) else None
    except Exception:  # 자료가 없으면 해당 제안만 건너뛴다
        return None


def _row(df: pd.DataFrame, label: str):
    r = df[df.구분 == label]
    return r.iloc[0] if len(r) else None


def build() -> list[Advice]:
    ph, pa, pv = _get(A.pedestrian_risk, "hour"), _get(A.pedestrian_risk, "victim_age"), _get(A.pedestrian_risk, "violation")
    hot, enf = _get(A.pedestrian_hotspots, None, 5), _get(A.enforcement_overview)
    al, ix = _get(A.risk_by_factor, "alignment"), _get(A.risk_by_factor, "intersection")
    wx = _get(A.risk_by_factor, "weather")
    sf = _get(A.risk_by_factor, "surface", wx.years[0] if wx is not None and wx.years else None)
    ch, cr = _get(A.child_risk, "hour"), _get(A.child_risk, "road")
    out: list[Advice] = []

    # ---- 담당 기관 --------------------------------------------------------------
    if hot is not None:
        ped = A._ped()
        total = int((ped.geo_ok == 1).sum())
        top = hot.df.head(5)
        n5 = int(top["사고 건수"].sum())
        names = [str(x).split(". ", 1)[-1] for x in top["장소"]]
        out.append(Advice(
            "admin", "보행자 사고 다발 지점 우선 점검",
            f"보행자 사고가 같은 지점(약 400m)에서 많이 난 상위 {len(top)}곳이 전체 {total:,}건의 {n5 / total * 100:.0f}%({n5}건)를 차지합니다. "
            f"1위는 '{names[0]}'({int(top.iloc[0]['사고 건수'])}건)입니다.",
            "이 지점들을 현장 점검 대상으로 우선 지정해 신호 주기, 횡단보도·정지선 시인성, 차량 접근 속도, 불법 주정차 여부를 확인하는 것을 검토합니다.",
            "약 400m 격자 단위라 정확한 위치는 현장 확인이 필요하고, 보행량이 많아서 사고가 많은 곳일 수 있습니다.",
            _strength(n5, False), "보행자 사고 다발 지점", items=[f"{n} ({int(c)}건)" for n, c in zip(names, top["사고 건수"])],
            map_df=hot.map_df, map_legend=hot.map_legend))

    if ph is not None:
        t3 = ph.df.sort_values("사고 건수", ascending=False).head(3)
        n3, share = int(t3["사고 건수"].sum()), float(t3["구성비(%)"].sum())
        out.append(Advice(
            "admin", "보행자 사고 집중 시간대 점검",
            f"보행자 사고의 {share:.0f}%({n3}건)가 {', '.join(t3.구분)}에 몰렸습니다(전체 {int(ph.df['사고 건수'].sum()):,}건).",
            "해당 시간대에 횡단보도 주변 계도·순찰을 늘리고, 조명과 횡단보도 시인성을 점검하는 것을 검토합니다.",
            "이 시간대는 보행·교통량이 많은 시간일 수 있어, 건수만으로 위험이 더 크다고 말할 수 없습니다.",
            _strength(n3, True), "보행자 사고 시간대별"))

    if pa is not None and pa.df["사망자"].sum() > 0:
        old = pa.df[pa.df.구분.isin(_OLD)]
        sv, sd = float(old["구성비(%)"].sum()), float(old["사망자"].sum() / pa.df["사망자"].sum() * 100)
        if sd >= sv * 1.5:
            out.append(Advice(
                "admin", "고령 보행자 보호 점검",
                f"60대 이상이 보행자 사고 피해자의 {sv:.0f}%인데 사망자의 {sd:.0f}%({int(old['사망자'].sum())}명)를 차지합니다.",
                "고령 보행자가 많은 다발 지점의 횡단 신호 시간, 중앙 보행섬, 야간 시인성 확보를 점검하는 것을 검토합니다.",
                "연령대별 보행 인구가 반영되지 않았고, 고령자는 사고 시 사망으로 이어질 가능성이 높아 치명도가 높게 나올 수 있습니다.",
                _strength(int(old["사고 건수"].sum()), True), "보행자 사고 피해자 연령대별"))

    if pv is not None and len(pv.df) >= 2 and enf is not None:
        v1, v2, e1 = pv.df.iloc[0], pv.df.iloc[1], enf.df.iloc[0]
        out.append(Advice(
            "admin", "보행자 보호 관련 단속·계도 비중 검토",
            f"보행자 사고의 가해 운전자 법규위반은 '{v1.구분}'({v1['구성비(%)']}%)과 '{v2.구분}'({v2['구성비(%)']}%)이 대부분인데, "
            f"교통단속은 '{e1['위반 유형']}'이 {e1['비중(%)']}%를 차지합니다.",
            "보행자 보호의무·안전운전 관련 계도와 단속의 비중이 현재 사고 원인과 맞는지 점검하는 것을 검토합니다.",
            "단속 자료는 도로 전체의 단속이고 보행자 사고 자료와 대상이 달라 직접 비교하기 어렵습니다.",
            "약함", "보행자 사고 법규위반 · 법규위반 단속"))

    if al is not None:
        base = _row(al.df, "직선 평지")
        cand = al.df[(al.df["사고 건수"] >= 30) & ~al.df.구분.isin(["직선 평지"]) & ~al.df.구분.str.contains("기타")]
        if base is not None and base.치명도 > 0 and len(cand):
            t = cand.sort_values("치명도", ascending=False).iloc[0]
            r = t.치명도 / base.치명도
            if r >= 1.5:
                out.append(Advice(
                    "admin", f"'{t.구분}' 구간 점검",
                    f"'{t.구분}' 구간의 치명도는 직선 평지의 {r:.1f}배입니다({t.치명도}명 vs {base.치명도}명, 사고 {int(t['사고 건수'])}건).",
                    "이런 형태의 도로 구간에서 속도 저감 시설, 노면 표지·시선 유도 시설, 미끄럼 방지 포장 상태를 점검하는 것을 검토합니다.",
                    "사고 수가 적어 연도별로 크게 흔들릴 수 있고, 구간별 통행량은 반영되지 않았습니다.",
                    _strength(int(t["사고 건수"]), True), "도로선형별 사고"))

    if ix is not None:
        total = int(ix.df["사고 건수"].sum())
        cross = ix.df[ix.df.구분 != "교차로 아님"]
        if len(cross) and total:
            t = cross.sort_values("사고 건수", ascending=False).iloc[0]
            sh = t["사고 건수"] / total * 100
            if sh >= 25:
                out.append(Advice(
                    "admin", f"{t.구분} 교차로 신호·시거 점검",
                    f"교차로 형태 중 '{t.구분}' 교차로에서 전체 사고의 {sh:.0f}%({int(t['사고 건수']):,}건)가 발생했습니다.",
                    "사고가 많은 형태의 교차로에서 신호 운영(좌회전·보행 신호 중첩), 시거 확보, 정지선 위치를 점검하는 것을 검토합니다.",
                    "교차로 수와 통행량이 반영되지 않아 교차로 한 곳당 위험도는 알 수 없습니다.",
                    _strength(int(t["사고 건수"]), False), "교차로 형태별 사고"))

    rw = _row(wx.df, "비") if wx is not None else None
    bw = _row(wx.df, "맑음") if wx is not None else None
    rs = _row(sf.df, "젖음/습기") if sf is not None else None
    bs = _row(sf.df, "건조") if sf is not None else None
    wet = None
    if rw is not None and bw is not None and rs is not None and bs is not None and bw.치명도 > 0 and bs.치명도 > 0:
        r1, r2 = rw.치명도 / bw.치명도, rs.치명도 / bs.치명도
        if r1 >= 1.3 and r2 >= 1.3:
            wet = (r1, r2, int(min(rw["사고 건수"], rs["사고 건수"])))
            out.append(Advice(
                "admin", "우천 시 속도 안내·노면 점검",
                f"비 오는 날의 치명도는 맑은 날의 {r1:.1f}배, 젖은 노면은 건조 노면의 {r2:.1f}배입니다.",
                "우천 시 사고 위험 구간에 속도 안내(전광판·내비게이션 알림)를 확대하고, 배수 불량·미끄럼에 취약한 구간의 포장·배수 상태를 점검하는 것을 검토합니다.",
                "교통량이 반영되지 않았고, 비 오는 날은 사고 수 자체가 적어 값이 흔들릴 수 있습니다.",
                _strength(wet[2], True), "기상 · 노면 상태별 사고"))

    if ch is not None and cr is not None:
        t3 = ch.df.sort_values("사고 건수", ascending=False).head(3)
        r1 = cr.df.iloc[0]
        out.append(Advice(
            "admin", "통학 시간대·지방도 어린이 보행 안전 점검",
            f"12세 이하 어린이 사고의 {t3['구성비(%)'].sum():.0f}%가 {', '.join(t3.구분)}에 몰렸고, {r1['구성비(%)']}%가 '{r1.구분}'에서 발생했습니다.",
            "통학로가 지나는 지방도 구간의 보도·횡단보도·과속 방지 시설과 하교 시간대 계도를 점검하는 것을 검토합니다.",
            "도로 종류별 연장과 어린이 통행량이 반영되지 않았습니다(연장이 긴 도로는 사고 비중이 높게 나올 수 있습니다).",
            _strength(int(ch.df["사고 건수"].sum()), True), "어린이 교통사고"))

    # ---- 도민 --------------------------------------------------------------------
    if wet:
        out.append(Advice(
            "citizen", "비 오는 날에는 평소보다 속도를 줄여요",
            f"비 오는 날의 사고 치명도는 맑은 날의 {wet[0]:.1f}배, 젖은 노면은 건조 노면의 {wet[1]:.1f}배입니다.",
            "비가 오거나 길이 젖어 있으면 속도를 줄이고 앞차와 거리를 더 두세요.",
            "교통량이 반영되지 않은 비교입니다. 사고가 적은 날씨라 값이 흔들릴 수 있습니다.", _strength(wet[2], True), "기상 · 노면 상태별 사고"))
    if ph is not None:
        t3 = ph.df.sort_values("사고 건수", ascending=False).head(3)
        out.append(Advice(
            "citizen", "해질녘 횡단보도에서는 한 번 더 살펴요",
            f"보행자 사고의 {t3['구성비(%)'].sum():.0f}%가 {', '.join(t3.구분)}에 몰렸습니다.",
            "이 시간대에 길을 건널 때는 차가 멈춘 것을 확인하고 건너며, 밝은 옷이나 반사 용품을 착용하세요. 운전자는 횡단보도 앞에서 속도를 줄이세요.",
            "시간대 집중은 확인됐지만, 밝은 옷·반사 용품의 효과 자체는 이 데이터로 확인할 수 없는 일반 안전 수칙입니다.",
            _strength(int(t3["사고 건수"].sum()), True), "보행자 사고 시간대별"))
    if pa is not None and pa.df["사망자"].sum() > 0:
        old = pa.df[pa.df.구분.isin(_OLD)]
        sd = float(old["사망자"].sum() / pa.df["사망자"].sum() * 100)
        if sd >= 40:
            out.append(Advice(
                "citizen", "부모님·어르신께 횡단 요령을 알려 드려요",
                f"보행자 사고 사망자의 {sd:.0f}%가 60대 이상입니다.",
                "신호가 바뀐 뒤에도 좌우를 한 번 더 살피고, 시간이 부족하면 무리하지 말고 다음 신호를 기다리도록 알려 드리세요.",
                "연령대별 보행 인구가 반영되지 않았고, 구체적인 행동 요령은 일반 안전 수칙입니다.",
                _strength(int(old["사고 건수"].sum()), True), "보행자 사고 피해자 연령대별"))
    if ch is not None:
        t1 = ch.df.sort_values("사고 건수", ascending=False).iloc[0]
        out.append(Advice(
            "citizen", "하교 시간에는 통학로에서 서행해요",
            f"12세 이하 어린이 사고가 가장 많은 시간대는 {t1.구분}({t1['구성비(%)']}%)입니다.",
            "이 시간대에 학교 주변과 통학로를 지날 때는 속도를 줄이고, 보호자는 아이가 건널 때 함께 확인해 주세요.",
            "어린이 통행량이 반영되지 않았고, 보호자 동행의 효과는 이 데이터로 확인할 수 없는 일반 안전 수칙입니다.",
            _strength(int(ch.df["사고 건수"].sum()), True), "어린이 교통사고"))
    if pv is not None:
        v1 = pv.df.iloc[0]
        out.append(Advice(
            "citizen", "횡단보도 앞에서는 보행자가 없어도 속도를 줄여요",
            f"보행자 사고에서 가해 운전자의 법규위반은 '{v1.구분}'이 {v1['구성비(%)']}%로 가장 많았습니다.",
            "횡단보도와 교차로 앞에서는 보행자가 보이지 않아도 속도를 줄이고 주변을 살피세요.",
            "법규위반 유형은 사고 당시의 경찰 기록 기준이며 과실 비율을 뜻하지 않습니다.",
            _strength(int(v1["사고 건수"]), False), "보행자 사고 법규위반"))
    if al is not None:
        base = _row(al.df, "직선 평지")
        cand = al.df[(al.df["사고 건수"] >= 30) & ~al.df.구분.isin(["직선 평지"]) & ~al.df.구분.str.contains("기타")]
        if base is not None and base.치명도 > 0 and len(cand):
            t = cand.sort_values("치명도", ascending=False).iloc[0]
            if t.치명도 / base.치명도 >= 1.5:
                out.append(Advice(
                    "citizen", "커브길 내리막은 들어가기 전에 감속해요",
                    f"'{t.구분}' 구간의 치명도는 직선 평지의 {t.치명도 / base.치명도:.1f}배입니다(사고 {int(t['사고 건수'])}건).",
                    "커브가 이어지는 내리막에서는 진입 전에 미리 속도를 줄이세요.",
                    "사고 수가 적어 값이 흔들릴 수 있습니다.", _strength(int(t["사고 건수"]), True), "도로선형별 사고"))

    for a in out:
        a.art = _art_for(a.title)
    out.sort(key=lambda a: (a.audience != "admin", _RANK[a.strength]))  # 담당 기관 먼저, 근거 강한 순
    return out


def to_frame(items: list[Advice]) -> pd.DataFrame:
    return pd.DataFrame([{"대상": "담당 기관" if a.audience == "admin" else "도민", "제안": a.title, "내용": a.action, "근거": a.basis,
                          "신뢰": f"근거 {a.strength}", "한계": a.caveat} for a in items])
