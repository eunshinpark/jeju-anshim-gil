"""도로선형 그림: 직선·좌커브·우커브 × 평지·오르막·내리막을 도로 모양으로 그리고 사고 수치를 붙인 카드.

순수 문자열(HTML+SVG)만 만들기 때문에 Streamlit 화면과 내려받는 HTML 보고서에서 똑같이 쓴다.
도로는 위에서 내려다본 모양, 오른쪽 아래 작은 상자는 옆에서 본 경사(평지/오르막/내리막)다.
"""
from __future__ import annotations

import html
import math

import pandas as pd

ROAD, EDGE, ACC, MUTED = "#57606a", "#e1e1e1", "#224da0", "#666666"

# 위에서 본 도로 중심선 (viewBox 0 0 120 90)
_PATHS = {
    "직선": "M60,84 L60,6",
    "좌커브": "M64,84 L64,44 Q64,22 42,22 L10,22",
    "우커브": "M56,84 L56,44 Q56,22 78,22 L110,22",
}
# 옆에서 본 경사: 작은 상자(76..114, 56..82) 안의 선
_GRADE = {"평지": "M82,69 L108,69", "오르막": "M82,77 L108,61", "내리막": "M82,61 L108,77"}


def _svg(shape: str, slope: str) -> str:
    road = _PATHS[shape]
    inset = (f"<rect x='74' y='54' width='42' height='30' rx='5' fill='#f6f8fa' stroke='{EDGE}'/>"
             f"<path d='{_GRADE[slope]}' stroke='{ACC}' stroke-width='3' stroke-linecap='round' fill='none'/>")
    return (f"<svg viewBox='0 0 120 90' width='120' height='90' role='img' aria-label='{html.escape(shape)} {html.escape(slope)}'>"
            f"<path d='{road}' stroke='{EDGE}' stroke-width='20' stroke-linecap='round' stroke-linejoin='round' fill='none'/>"
            f"<path d='{road}' stroke='{ROAD}' stroke-width='16' stroke-linecap='round' stroke-linejoin='round' fill='none'/>"
            f"<path d='{road}' stroke='#ffffff' stroke-width='1.6' stroke-dasharray='5 5' fill='none'/>{inset}</svg>")


def _svg_other() -> str:
    return ("<svg viewBox='0 0 120 90' width='120' height='90' role='img' aria-label='기타구역'>"
            f"<path d='M28,84 L28,6' stroke='{EDGE}' stroke-width='20' stroke-linecap='round' fill='none'/>"
            f"<path d='M28,84 L28,6' stroke='{ROAD}' stroke-width='16' stroke-linecap='round' fill='none'/>"
            "<path d='M28,84 L28,6' stroke='#ffffff' stroke-width='1.6' stroke-dasharray='5 5' fill='none'/>"
            f"<rect x='52' y='16' width='56' height='56' rx='6' fill='#eaeef2' stroke='{EDGE}'/>"
            f"<text x='80' y='52' font-size='26' font-weight='700' text-anchor='middle' fill='{MUTED}'>P</text></svg>")


def _kind(label: str):
    """'커브,곡각 좌 내리막' -> ('좌커브', '내리막')"""
    slope = next((s for s in ("오르막", "내리막", "평지") if s in label), None)
    if "기타구역" in label:
        return "기타", None
    if "커브" in label:
        return ("좌커브" if " 좌 " in f" {label} " else "우커브"), slope
    return "직선", slope


_CSS = """<style>
.ad{margin:.4rem 0 .2rem}
.ad-g{display:grid;grid-template-columns:70px repeat(3,minmax(0,1fr));gap:10px;align-items:stretch}
.ad-h{font-size:12px;color:#666666;text-align:center;font-weight:600;padding-bottom:2px}
.ad-r{display:flex;align-items:center;font-size:13px;font-weight:600;color:#333333}
.ad-c{border:1px solid #e1e1e1;border-radius:6px;background:#fff;padding:8px 6px 10px;text-align:center;position:relative}
.ad-c svg{display:block;margin:0 auto 4px;max-width:100%;height:auto}
.ad-c.hot{border-color:#ff5165;box-shadow:0 0 0 1px #ff5165 inset}
.ad-c.dim{opacity:.55}
.ad-n{font-size:20px;font-weight:600;line-height:1.15}.ad-n span{font-size:12px;font-weight:400;color:#666666;margin-left:2px}
.ad-s{font-size:12px;color:#666666;margin-top:2px}
.ad-b{height:4px;border-radius:2px;background:#eaeef2;margin:6px 10px 0;overflow:hidden}.ad-b i{display:block;height:100%;background:#224da0}
.ad-tag{position:absolute;top:6px;left:6px;font-size:11px;line-height:16px;padding:0 6px;border-radius:2em;background:#ffe9ec;color:#d6203a;border:1px solid rgba(255,129,130,.4)}
.ad-note{font-size:12px;color:#666666;margin-top:8px}
@media(max-width:700px){.ad-g{grid-template-columns:52px repeat(3,minmax(0,1fr))}}
</style>"""


def _card(svg: str, row: pd.Series | None, mx: float, hot: bool, name: str = "") -> str:
    if row is None:
        return "<div class='ad-c dim'>" + svg + "<div class='ad-s'>자료 없음</div></div>"
    n = int(row["사고 건수"])
    cls = "ad-c" + (" hot" if hot else "") + (" dim" if n < 30 else "")
    tag = "<span class='ad-tag'>치명도 최고</span>" if hot else ""
    note = " · 표본 적음" if n < 30 else ""
    title = f"<div class='ad-t'>{html.escape(name)}</div>" if name else ""
    return (f"<div class='{cls}'>{tag}{svg}{title}<div class='ad-n'>{n:,}<span>건</span></div>"
            f"<div class='ad-s'>치명도 {row['치명도']} · 사망 {int(row['사망자'])}명{note}</div>"
            f"<div class='ad-b'><i style='width:{max(n / mx * 100, 2):.1f}%'></i></div></div>")


def alignment_cards(df: pd.DataFrame) -> str:
    """도로선형별 결과표(구분, 사고 건수, 사망자, 치명도)를 도로 그림 카드로 만든다."""
    need = {"구분", "사고 건수", "사망자", "치명도"}
    if df is None or df.empty or not need <= set(df.columns):
        return ""
    cells = {}
    for _, r in df.iterrows():
        k = _kind(str(r["구분"]))
        cells[k] = r
    mx = float(df["사고 건수"].max()) or 1.0
    # 치명도 최고 칸: 사고 30건 이상인 칸 중에서만 고른다(표본이 적으면 우연일 수 있다)
    solid = [(k, r) for k, r in cells.items() if k[0] != "기타" and r["사고 건수"] >= 30]
    hot_key = max(solid, key=lambda kr: kr[1]["치명도"])[0] if solid else None

    out = ["<div class='ad'>", _CSS, "<div class='ad-g'><div></div>"]
    out += [f"<div class='ad-h'>{s}</div>" for s in ("평지", "오르막", "내리막")]
    for shape, name in (("직선", "직선"), ("좌커브", "좌측 커브"), ("우커브", "우측 커브")):
        out.append(f"<div class='ad-r'>{name}</div>")
        for slope in ("평지", "오르막", "내리막"):
            k = (shape, slope)
            out.append(_card(_svg(shape, slope), cells.get(k), mx, k == hot_key))
    if ("기타", None) in cells:
        out.append("<div class='ad-r'>기타</div>")
        out.append(_card(_svg_other(), cells[("기타", None)], mx, False))
    out.append("</div>")
    out.append("<div class='ad-note'>그림은 도로를 위에서 내려다본 모양이고, 오른쪽 아래 작은 상자는 옆에서 본 경사입니다. "
               "막대는 가장 사고가 많은 칸 대비 비율이며, 사고 30건 미만 칸은 흐리게 표시했습니다.</div></div>")
    return "".join(out)


# ----------------------------------------------------------------------------- 교차로 그림
_CX, _CY, _OUT = 60, 45, 34
_X_ARMS = {  # 중심에서 뻗어 나가는 도로의 방향(도). 90=위, 0=오른쪽, 270=아래, 180=왼쪽
    "삼지": [180, 0, 270],
    "사지": [90, 0, 270, 180],
    "오지이상": [90, 162, 234, 306, 18],
    "회전": [90, 0, 270, 180],
    "교차로 아님": [90, 270],
}
_X_NAMES = {"삼지": "삼지 교차로", "사지": "사지 교차로", "오지이상": "오지 이상", "회전": "회전 교차로", "교차로 아님": "교차로 아님"}
_CSS_X = ("<style>.ad-g5{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}"
          ".ad-t{font-size:13px;font-weight:600;margin-bottom:2px}"
          "@media(max-width:860px){.ad-g5{grid-template-columns:repeat(3,minmax(0,1fr))}}</style>")


def _arm(angle: float, r0: float, r1: float) -> str:
    a = math.radians(angle)
    return (f"M{_CX + r0 * math.cos(a):.1f},{_CY - r0 * math.sin(a):.1f} "
            f"L{_CX + r1 * math.cos(a):.1f},{_CY - r1 * math.sin(a):.1f}")


def _xsvg(kind: str) -> str:
    """교차로를 위에서 내려다본 그림. 삼지=T자, 사지=十자, 오지이상=별 모양, 회전=원형 도로에 진입로 4개."""
    arms, ring = _X_ARMS[kind], kind == "회전"
    r0 = 15 if ring else 0
    edge = "".join(f"<path d='{_arm(a, r0, _OUT)}' stroke='{EDGE}' stroke-width='20' stroke-linecap='square' fill='none'/>" for a in arms)
    road = "".join(f"<path d='{_arm(a, r0, _OUT)}' stroke='{ROAD}' stroke-width='16' stroke-linecap='square' fill='none'/>" for a in arms)
    dash_from = 0 if kind == "교차로 아님" else (24 if not ring else 24)
    dash = "".join(f"<path d='{_arm(a, dash_from, _OUT)}' stroke='#ffffff' stroke-width='1.6' stroke-dasharray='5 5' fill='none'/>" for a in arms)
    island = ""
    if ring:
        edge += f"<circle cx='{_CX}' cy='{_CY}' r='15' fill='none' stroke='{EDGE}' stroke-width='20'/>"
        road += f"<circle cx='{_CX}' cy='{_CY}' r='15' fill='none' stroke='{ROAD}' stroke-width='16'/>"
        island = f"<circle cx='{_CX}' cy='{_CY}' r='7' fill='#eaeef2'/>"
    return (f"<svg viewBox='0 0 120 90' width='120' height='90' role='img' aria-label='{html.escape(_X_NAMES[kind])}'>"
            f"{edge}{road}{dash}{island}</svg>")


def intersection_cards(df: pd.DataFrame) -> str:
    """교차로 형태별 결과표(구분, 사고 건수, 사망자, 치명도)를 교차로 그림 카드로 만든다."""
    need = {"구분", "사고 건수", "사망자", "치명도"}
    if df is None or df.empty or not need <= set(df.columns):
        return ""
    rows = {str(r["구분"]): r for _, r in df.iterrows()}
    order = [k for k in ("삼지", "사지", "오지이상", "회전", "교차로 아님") if k in rows]
    if not order:
        return ""
    mx = float(df["사고 건수"].max()) or 1.0
    solid = [k for k in order if k != "교차로 아님" and rows[k]["사고 건수"] >= 30]
    hot = max(solid, key=lambda k: rows[k]["치명도"]) if solid else None
    cards = "".join(_card(_xsvg(k), rows[k], mx, k == hot, _X_NAMES[k]) for k in order)
    return ("<div class='ad'>" + _CSS + _CSS_X + f"<div class='ad-g5'>{cards}</div>"
            "<div class='ad-note'>그림은 교차로를 위에서 내려다본 모양입니다. 막대는 가장 사고가 많은 칸 대비 비율이고, "
            "'치명도 최고'는 교차로 형태 중 사고 30건 이상인 곳에서만 고릅니다.</div></div>")


# ----------------------------------------------------------------------------- 개선 제안 일러스트
# 사진 대신 직접 그린 단순한 일러스트(저작권 문제가 없고 인터넷 없이도 나온다). 모두 viewBox 0 0 120 90.
_SUN, _SKIN = "#f5b301", "#f4c7a1"


def _road(y: int = 62, stripes=()) -> str:
    s = (f"<rect x='0' y='{y}' width='120' height='{90 - y}' fill='{ROAD}'/>"
         f"<path d='M0 {y + (90 - y) / 2:.0f} H120' stroke='#fff' stroke-width='1.6' stroke-dasharray='6 5' fill='none'/>")
    return s + "".join(f"<rect x='{x}' y='{y}' width='6' height='{90 - y}' fill='#fff' opacity='.9'/>" for x in stripes)


def _person(x, y, h=30, shirt=ACC, pants="#3a4a6b", hair="#4a3b32", cane=False, pack=False, vest=False) -> str:
    """서 있는 사람. (x, y)는 발 위치, h는 키."""
    r, bw, bh = h * 0.13, h * 0.3, h * 0.38
    top = y - h
    by = top + 2 * r + 1
    ly = by + bh
    s = [f"<circle cx='{x}' cy='{top + r:.1f}' r='{r:.1f}' fill='{_SKIN}'/>",
         f"<path d='M{x - r:.1f},{top + r:.1f} a{r:.1f},{r:.1f} 0 0 1 {2 * r:.1f},0 z' fill='{hair}'/>",
         f"<path d='M{x - bw * .22:.1f},{ly:.1f} L{x - bw * .3:.1f},{y} M{x + bw * .22:.1f},{ly:.1f} L{x + bw * .3:.1f},{y}' "
         f"stroke='{pants}' stroke-width='{h * .07:.1f}' stroke-linecap='round'/>",
         f"<rect x='{x - bw / 2:.1f}' y='{by:.1f}' width='{bw:.1f}' height='{bh:.1f}' rx='{bw * .3:.1f}' fill='{shirt}'/>",
         f"<path d='M{x - bw / 2:.1f},{by + bh * .15:.1f} L{x - bw * .85:.1f},{by + bh * .8:.1f} M{x + bw / 2:.1f},{by + bh * .15:.1f} "
         f"L{x + bw * .85:.1f},{by + bh * .8:.1f}' stroke='{shirt}' stroke-width='{h * .06:.1f}' stroke-linecap='round'/>"]
    if vest:
        s += [f"<rect x='{x - bw / 2:.1f}' y='{by + bh * .3:.1f}' width='{bw:.1f}' height='{bh * .13:.1f}' fill='#fff'/>",
              f"<rect x='{x - bw / 2:.1f}' y='{by + bh * .62:.1f}' width='{bw:.1f}' height='{bh * .13:.1f}' fill='#fff'/>"]
    if cane:
        s.append(f"<path d='M{x + bw * .85:.1f},{by + bh * .8:.1f} L{x + bw * 1.15:.1f},{y}' stroke='#8a6d3b' stroke-width='1.8' stroke-linecap='round'/>")
    if pack:
        s.append(f"<rect x='{x - bw / 2 - bw * .3:.1f}' y='{by:.1f}' width='{bw * .34:.1f}' height='{bh * .8:.1f}' rx='2' fill='#ff5165'/>")
    return "".join(s)


def _car(x, y, w=42, color="#ff5165") -> str:
    """자동차(옆모습). (x, y)는 왼쪽 바퀴 중심 높이."""
    h = w * .4
    return (f"<rect x='{x}' y='{y - h * .6:.1f}' width='{w}' height='{h * .6:.1f}' rx='4' fill='{color}'/>"
            f"<path d='M{x + w * .2:.1f},{y - h * .6:.1f} L{x + w * .32:.1f},{y - h:.1f} L{x + w * .7:.1f},{y - h:.1f} L{x + w * .84:.1f},{y - h * .6:.1f} Z' fill='{color}'/>"
            f"<path d='M{x + w * .32:.1f},{y - h * .62:.1f} L{x + w * .38:.1f},{y - h * .93:.1f} L{x + w * .66:.1f},{y - h * .93:.1f} L{x + w * .74:.1f},{y - h * .62:.1f} Z' fill='#cfe3f7'/>"
            f"<circle cx='{x + w * .25:.1f}' cy='{y}' r='{h * .26:.1f}' fill='#2b2f36'/><circle cx='{x + w * .75:.1f}' cy='{y}' r='{h * .26:.1f}' fill='#2b2f36'/>")


def _rays(cx, cy, r0, r1, angles) -> str:
    return "".join(f"<path d='M{cx + r0 * math.cos(math.radians(a)):.1f},{cy - r0 * math.sin(math.radians(a)):.1f} "
                   f"L{cx + r1 * math.cos(math.radians(a)):.1f},{cy - r1 * math.sin(math.radians(a)):.1f}' stroke='{_SUN}' stroke-width='2.2' stroke-linecap='round'/>"
                   for a in angles)


def _signal(x) -> str:
    return (f"<rect x='{x + 2}' y='28' width='3' height='36' fill='#666'/><rect x='{x - 5}' y='12' width='17' height='24' rx='3' fill='#333'/>"
            f"<circle cx='{x + 3.5}' cy='19' r='4' fill='#555'/><circle cx='{x + 3.5}' cy='29' r='4' fill='#2fbf71'/>")


def _art(key: str) -> str:
    if key == "intersection":
        return _xsvg("사지")
    if key == "curve_down":
        return _svg("좌커브", "내리막")
    if key == "hotspot":
        body = (f"<path d='M60,6 c-13,0 -22,10 -22,22 c0,15 22,34 22,34 s22,-19 22,-34 c0,-12 -9,-22 -22,-22 z' fill='#ff5165'/>"
                f"<circle cx='60' cy='28' r='8' fill='#fff'/>" + _road(66, (22, 34, 46, 58, 70, 82, 94)))
    elif key == "peak_hours":
        body = (f"<circle cx='32' cy='54' r='20' fill='{_SUN}'/>" + _rays(32, 54, 24, 31, (200, 235, 270, 305, 340)) + _road(60)
                + f"<circle cx='90' cy='30' r='19' fill='#fff' stroke='{ACC}' stroke-width='3'/>"
                f"<path d='M90 30 V19 M90 30 L98 35' stroke='{ACC}' stroke-width='2.4' stroke-linecap='round' fill='none'/>"
                + _person(62, 84, 28))
    elif key == "sunset_cross":
        body = (f"<circle cx='30' cy='56' r='22' fill='#ff9f43'/>" + _rays(30, 56, 26, 33, (205, 240, 270, 300, 335)) + _road(62, (56, 68, 80, 92))
                + _person(74, 86, 32, shirt=_SUN, vest=True))
    elif key == "elderly":
        body = (_road(64, (22, 34, 46, 58, 70, 82)) + _person(40, 86, 34, shirt="#6b8e7f", hair="#d9d9d9", cane=True)
                + _person(64, 86, 36) + _signal(98))
    elif key == "enforcement":
        body = (_road(64) + "<rect x='86' y='20' width='5' height='46' fill='#666'/>"
                f"<rect x='64' y='12' width='38' height='19' rx='4' fill='{ACC}'/>"
                "<circle cx='75' cy='21.5' r='6.5' fill='#cfe3f7' stroke='#fff' stroke-width='1.6'/><circle cx='96' cy='17' r='2.5' fill='#ff5165'/>"
                f"<path d='M70 28 L38 62' stroke='{_SUN}' stroke-width='1.6' stroke-dasharray='3 3' fill='none'/>" + _car(10, 84, 44))
    elif key == "rain":
        cloud = "".join(f"<ellipse cx='{cx}' cy='{cy}' rx='{rx}' ry='{ry}' fill='#9fb0c8'/>"
                        for cx, cy, rx, ry in ((42, 22, 20, 11), (60, 17, 16, 12), (78, 23, 17, 9)))
        drops = "".join(f"<path d='M{x},{y} l-3,9' stroke='#00a1e9' stroke-width='2' stroke-linecap='round'/>" for x, y in
                        ((34, 38), (46, 42), (58, 36), (70, 42), (82, 37), (40, 52), (64, 52), (88, 50)))
        body = cloud + drops + _road(66) + _car(34, 85, 46)
    elif key == "child":
        body = (f"<rect x='62' y='26' width='46' height='36' fill='#ffd666'/><path d='M58,28 L85,10 L112,28 Z' fill='{ACC}'/>"
                "<rect x='80' y='42' width='10' height='20' fill='#8a6d3b'/><rect x='68' y='34' width='8' height='8' fill='#cfe3f7'/>"
                "<rect x='96' y='34' width='8' height='8' fill='#cfe3f7'/>" + _road(64, (14, 26, 38, 50)) + _person(30, 86, 28, shirt="#ff5165", pack=True))
    elif key == "crosswalk_slow":
        body = (_road(62, (72, 84, 96)) + "<rect x='103' y='28' width='2' height='34' fill='#666'/>"
                f"<rect x='96' y='12' width='16' height='16' rx='2' fill='{ACC}'/><path d='M104 15 l-5 10 h10 z' fill='#fff'/>"
                + _person(86, 86, 30) + _car(8, 84, 44) + "<path d='M2 66 H8 M0 72 H6' stroke='#fff' stroke-width='1.6' opacity='.7'/>")
    else:
        return ""
    return f"<svg viewBox='0 0 120 90' role='img' aria-label='{html.escape(key)}'>{body}</svg>"


def advice_art(key: str) -> str:
    """개선 제안 카드 옆에 붙일 일러스트(SVG 문자열). 없는 키는 빈 문자열."""
    try:
        return _art(key)
    except Exception:
        return ""


# ----------------------------------------------------------------------------- 보고서용 개선 제안 카드(일러스트 포함)
_CSS_ADV = ("<style>.av{margin:.3rem 0}.av-card{display:grid;grid-template-columns:150px 1fr;gap:18px;border:1px solid #e1e1e1;border-radius:8px;"
            "background:#fff;padding:14px 16px;margin:10px 0;break-inside:avoid}"
            ".av-art{background:#eaf0fa;border-radius:8px;padding:8px;display:flex;align-items:center;justify-content:center;align-self:start}"
            ".av-art svg{width:100%;height:auto;display:block}"
            ".av-h{font-weight:700;font-size:15px;line-height:1.5}"
            ".av-chip{display:inline-block;font-size:12px;font-weight:500;line-height:18px;padding:0 8px;border-radius:2em;margin-right:8px;border:1px solid}"
            ".av-chip.s{color:#224da0;border-color:#cddaf4;background:#eaf0fa}.av-chip.w{color:#9a6700;border-color:rgba(212,167,44,.4);background:#fff8c5}"
            ".av-chip.i{color:#666;border-color:#e1e1e1;background:#f4f6fb}"
            ".av-r{display:flex;gap:12px;font-size:14px;line-height:1.6;margin-top:8px}.av-r b{flex:none;width:34px;font-size:13px;color:#224da0}"
            ".av-c span{color:#666;font-size:13px}.av-pt{font-size:12.5px;color:#666;margin-top:8px}"
            "@media(max-width:700px){.av-card{grid-template-columns:1fr}.av-art{max-width:200px}}</style>")


def advice_cards(items) -> str:
    """개선 제안(advice.Advice 목록)을 일러스트가 붙은 카드 HTML 로 만든다. 보고서 화면과 내려받는 HTML 에서 쓴다."""
    if not items:
        return ""
    esc = html.escape
    cards = []
    for a in items:
        chip = {"강함": "s", "보통": "w", "약함": "i"}.get(a.strength, "i")
        art = advice_art(getattr(a, "art", ""))
        pts = f"<div class='av-pt'>우선 점검 지점 · {esc(' / '.join(a.items))}</div>" if getattr(a, "items", None) else ""
        body = (f"<div class='av-h'><span class='av-chip {chip}'>근거 {esc(a.strength)}</span>{esc(a.title)}</div>"
                f"<div class='av-r'><b>근거</b><span>{esc(a.basis)}</span></div>"
                f"<div class='av-r'><b>제안</b><span>{esc(a.action)}</span></div>"
                f"<div class='av-r av-c'><b>한계</b><span>{esc(a.caveat)}</span></div>{pts}")
        cards.append(f"<div class='av-card'><div class='av-art'>{art}</div><div>{body}</div></div>" if art
                     else f"<div class='av-card' style='grid-template-columns:1fr'><div>{body}</div></div>")
    return "<div class='av'>" + _CSS_ADV + "".join(cards) + "</div>"
