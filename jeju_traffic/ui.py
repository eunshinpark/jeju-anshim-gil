"""화면 부품: 스타일(CSS), 차트, 핵심 수치 카드. 화이트 모드 전용이며 GitHub Primer 디자인 토큰을 따른다.

차트 색은 검증된 팔레트(파랑 1색)를 쓰고, 가장 큰 값만 진하게 강조한다(나머지는 같은 색의 연한 단계).
텍스트는 색을 입히지 않고 보조 회색(muted)으로 둔다.
"""
from __future__ import annotations

import base64
import functools
import html
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .config import ROOT

LIGHT = dict(accent="#224da0", soft="#cddaf4", mid="#00a1e9", grid="#ececec", axis="#d9d9d9", muted="#666666")
DARK = dict(accent="#2f81f7", soft="#1f6feb", mid="#388bfd", grid="#21262d", axis="#30363d", muted="#8b949e")


def is_dark() -> bool:
    return False  # 화이트 모드 고정 (.streamlit/config.toml 의 base="light")


def pal() -> dict:
    return DARK if is_dark() else LIGHT


# 제주데이터허브(jejudatahub.net)의 색·글꼴·모양을 따른 스타일: 남색(#224da0) 강조, 연한 파랑 영역, 4px 모서리, #e1e1e1 테두리
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&family=Noto+Sans+KR:wght@400;500;700&display=swap');
:root { --accent:#224da0; --accent-dark:#1a3d82; --sky:#00a1e9; --fg:#333333; --muted:#666666; --border:#e1e1e1; --border-muted:#ececec;
        --subtle:#f4f6fb; --tint:#eaf0fa; --canvas:#ffffff; --success:#224da0; --danger:#ff5165; --attention:#9a6700; }
html, body, .stApp, .stMarkdown, .stText, p, h1, h2, h3, h4, label, button, input, textarea, [data-testid="stCaptionContainer"] {
  font-family: 'Noto Sans KR', 'Montserrat', 'Malgun Gothic', 'Apple SD Gothic Neo', -apple-system, 'Segoe UI', sans-serif !important;
}
.stApp { background: var(--canvas); color: var(--fg); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1012px; padding-top: .8rem; padding-bottom: 5rem; }
.stMarkdown p, .stMarkdown li { font-size: 15px; line-height: 1.6; }

/* 타이포그래피 */
h1 { font-size: 2rem !important; font-weight: 700 !important; color: var(--accent) !important; letter-spacing: 0 !important; line-height: 1.25 !important; margin: .2rem 0 .4rem !important; padding: 0 !important; }
h2, h3 { font-weight: 600 !important; letter-spacing: 0 !important; }
.hero { background: var(--tint); border: 1px solid #dbe5f6; border-radius: 8px; padding: 22px 28px 20px; margin: .6rem 0 1.1rem; }
.hero .eyebrow { margin: 0 0 .3rem; }
.hero h1 { font-size: 1.5rem !important; margin: .1rem 0 .45rem !important; }
.ldr { display: flex; align-items: center; gap: 14px; padding: 10px 2px; margin: .4rem 0; }
.ldr-dots { display: flex; gap: 6px; flex: none; }
.ldr-dots i { width: 10px; height: 10px; border-radius: 50%; display: block; animation: ldr-b 1.1s ease-in-out infinite; }
.ldr-dots i:nth-child(1) { background: #224da0; } .ldr-dots i:nth-child(2) { background: #00a1e9; animation-delay: .16s; } .ldr-dots i:nth-child(3) { background: #ff5165; animation-delay: .32s; }
@keyframes ldr-b { 0%, 70%, 100% { transform: translateY(0); opacity: .45; } 35% { transform: translateY(-7px); opacity: 1; } }
.ldr-t { font-weight: 600; color: var(--accent); font-size: 15px; } .ldr-s { font-size: 12.5px; color: var(--muted); margin-top: 2px; }
@media (prefers-reduced-motion: reduce) { .ldr-dots i { animation: none; opacity: .8; } }
.hero .lead { margin: 0; }
.eyebrow { display: inline-block; font-size: 12px; font-weight: 500; line-height: 18px; padding: 0 9px; border: 1px solid #cddaf4;
           border-radius: 2em; color: var(--accent); background: #fff; margin: 1.1rem 0 .3rem; }
.lead { font-size: 16px; color: var(--muted); margin: .2rem 0 1.3rem; line-height: 1.55; }
.sec { font-size: 17px; font-weight: 700; color: var(--accent); margin: 1.8rem 0 .8rem; padding-bottom: .3em; border-bottom: 1px solid var(--border-muted); }

/* 탭 (UnderlineNav) */
[data-testid="stButtonGroup"] [role="radiogroup"] { background: transparent !important; border: 0 !important; border-bottom: 1px solid var(--border-muted) !important;
  border-radius: 0 !important; gap: 4px !important; padding: 0 !important; width: 100% !important; }
button[data-variant="segmented_control"] { background: transparent !important; border: 0 !important; border-bottom: 2px solid transparent !important;
  border-radius: 4px !important; color: var(--fg) !important; font-size: 14px !important; font-weight: 400 !important; padding: 8px 14px !important;
  box-shadow: none !important; margin-bottom: -1px; }
button[data-variant="segmented_control"]:hover { background: rgba(34,77,160,.07) !important; }
button[data-variant="segmented_control"][aria-checked="true"] { font-weight: 700 !important; color: var(--accent) !important; border-bottom: 3px solid var(--accent) !important; border-radius: 4px 6px 0 0 !important; }

/* 왼쪽 메뉴(질문하기·자동 점검 보고서·데이터 현황): 박스 없이 글자만. 선택된 화면은 남색 굵은 글자 + 밑줄 */
.nav-h { display: none; }
[class*="st-key-nav_"] button { justify-content: flex-start !important; padding: 3px 12px !important; min-height: 0 !important; border-radius: 8px !important;
  font-size: 15.5px !important; font-weight: 500 !important; background: transparent !important; color: var(--fg) !important; border: 0 !important;
  box-shadow: none !important; transition: background .15s, color .15s !important; }
[class*="st-key-nav_"] button:hover { background: rgba(34,77,160,.08) !important; color: var(--accent) !important; }
[class*="st-key-nav_"] button[kind="primary"], [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] { background: transparent !important; color: var(--accent) !important;
  font-weight: 700 !important; box-shadow: none !important; }
[class*="st-key-nav_"] button[kind="primary"] *, [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] * { color: var(--accent) !important; font-weight: 700 !important; }
[class*="st-key-nav_"] button[kind="primary"] p, [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] p { text-decoration: underline !important;
  text-decoration-color: var(--accent) !important; text-decoration-thickness: 2px !important; text-underline-offset: 7px !important; }
[class*="st-key-nav_"] { margin-bottom: -.2rem; }
.st-key-nav_0 { margin-top: .9rem; }
[class*="st-key-nav_"] button > div, [class*="st-key-nav_"] button p, [class*="st-key-nav_"] button [data-testid="stMarkdownContainer"] { justify-content: flex-start !important; text-align: left !important; width: 100%; }
[class*="st-key-nav_"] button span[data-testid="stIconMaterial"] { font-size: 20px !important; margin-right: 6px; }

/* 맨 위 메뉴(질문하기·개선 제안…): 눈에 띄는 큰 버튼, 스크롤해도 맨 위에 붙어 있다 */
.st-key-page { position: sticky; top: 0; z-index: 99; background: #fff; padding: .55rem 0 .6rem; margin-bottom: .4rem; border-bottom: 1px solid var(--border-muted); }
.st-key-page [role="radiogroup"] { border-bottom: 0 !important; gap: 8px !important; }
.st-key-page button[data-variant="segmented_control"] { font-size: 16px !important; font-weight: 600 !important; padding: 10px 22px !important; border-radius: 999px !important;
  background: var(--tint) !important; color: var(--accent) !important; border: 1px solid #cddaf4 !important; margin-bottom: 0; transition: background .15s, box-shadow .15s; }
.st-key-page button[data-variant="segmented_control"]:hover { background: #dbe5f6 !important; }
.st-key-page button[data-variant="segmented_control"][aria-checked="true"] { background: var(--accent) !important; color: #fff !important; border-color: var(--accent) !important;
  box-shadow: 0 3px 10px rgba(34,77,160,.28) !important; }
.st-key-page button[data-variant="segmented_control"][aria-checked="true"] * { color: #fff !important; }

/* 핵심 수치 카드 */
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: .6rem 0 .4rem; }
.kpi { border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px; background: var(--canvas); }
.kpi .l { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.kpi .v { font-size: 30px; font-weight: 700; line-height: 1.2; color: var(--accent); font-family: 'Montserrat', 'Noto Sans KR', sans-serif !important; }
.kpi .v small { font-size: 13px; font-weight: 400; color: var(--muted); margin-left: 3px; }
.kpi .s { font-size: 12px; color: var(--muted); margin-top: 6px; }
@media (max-width: 860px) { .kpis { grid-template-columns: repeat(2, 1fr); } }

/* 박스 · 답변 */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 8px !important; border-color: var(--border) !important; background: var(--canvas); }
.rt { font-weight: 600; font-size: 14px; margin: .1rem 0 .5rem; }
.ct { font-size: 12px; color: var(--muted); margin: .3rem 0 -.2rem; }
.note { font-size: 13px; line-height: 1.55; color: var(--muted); border-left: 4px solid var(--border); padding: 0 1em; margin: .7rem 0 .9rem; }
.badge { display: inline-flex; align-items: center; justify-content: center; min-width: 20px; height: 20px; padding: 0 6px; border-radius: 2em;
         background: rgba(175,184,193,.2); color: var(--fg); font-size: 12px; font-weight: 500; margin-right: 8px; vertical-align: 1px; }
.find-h { font-weight: 600; font-size: 15px; line-height: 1.5; }
.find-e { font-size: 13px; color: var(--muted); margin-top: 4px; }
.find-c { font-size: 12px; color: var(--attention); margin-top: 3px; }

/* 버튼 · 입력 */
.stButton > button, .stDownloadButton > button { background: var(--canvas); color: var(--fg); border: 1px solid #d5d5d5; border-radius: 4px;
  font-size: 14px; font-weight: 500; box-shadow: none; padding: .4rem 1rem; text-align: left; justify-content: flex-start;
  transition: background .1s; }
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--subtle); border-color: var(--accent); color: var(--accent); }
.stButton > button[kind="primary"] { background: var(--accent); color: #fff; border-color: var(--accent); justify-content: center; font-weight: 700; }
.stButton > button[kind="primary"]:hover { background: var(--accent-dark); border-color: var(--accent-dark); color: #fff; }
.stButton > button p { text-align: left; width: 100%; }
.stButton > button[kind="primary"] p { text-align: center; }
[data-testid="stChatInput"] { border-radius: 4px; border-color: var(--border); }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(34,77,160,.22); }
[data-testid="stChatMessage"] { background: transparent; padding: .5rem 0; }
[data-testid="stChatMessageAvatarUser"] { background: #e4eaf6 !important; color: var(--accent) !important; }
[data-testid="stChatMessageAvatarAssistant"] { background: var(--accent) !important; color: #fff !important; }

/* 사이드바 */
.side-h { font-size: 12px; font-weight: 600; color: var(--muted); margin: .55rem 0 .55rem; }
[class*="st-key-h"] { margin-bottom: -.15rem; }
/* 질문 기록: 박스 없이 글자만, 왼쪽 정렬 */
[class*="st-key-h"] button { background: transparent !important; border: 0 !important; box-shadow: none !important; justify-content: flex-start !important;
  padding: 2px 8px !important; min-height: 0 !important; color: var(--fg) !important; font-weight: 400 !important; border-radius: 8px !important; }
[class*="st-key-h"] button:hover { background: rgba(34,77,160,.08) !important; color: var(--accent) !important; }
[class*="st-key-h"] button > div, [class*="st-key-h"] button p, [class*="st-key-h"] button [data-testid="stMarkdownContainer"] { justify-content: flex-start !important; text-align: left !important; }
[data-testid="stSidebar"] .stButton > button { background: transparent; border: 0; box-shadow: none; font-size: 13px; font-weight: 400;
  padding: .25rem .5rem; color: var(--fg); }
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(34,77,160,.08); color: var(--accent); border: 0; }
[data-testid="stSidebar"] { background: var(--subtle); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
/* 사이드바 간격: 기본 1rem 을 .7rem 으로(−30%), 구분선 여백도 줄인다 */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .7rem !important; }
[data-testid="stSidebar"] hr { margin: .5rem 0 !important; }
/* 맨 위 '제주 안심길'을 위로 올린다 */
[data-testid="stSidebarHeader"] { height: 1.6rem !important; min-height: 0 !important; padding-bottom: 0 !important; }
[data-testid="stSidebarUserContent"] { padding-top: .4rem !important; }
/* AI 설정: 박스 없이 글자만(메뉴와 같은 결). 펼치면 안쪽 입력칸만 보인다 */
[data-testid="stSidebar"] [data-testid="stExpander"] details { border: 0 !important; background: transparent !important; box-shadow: none !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary { padding: 3px 8px !important; border-radius: 8px !important; font-size: 14.5px !important; font-weight: 500 !important; color: var(--fg) !important; min-height: 0 !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background: rgba(34,77,160,.08) !important; color: var(--accent) !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: .3rem 0 0 8px !important; }
.brand { font-weight: 700; font-size: 28px; line-height: 1.2; letter-spacing: -0.02em; color: var(--accent); }
.brand .ms { font-size: 1.05em; margin-right: .3em; }
.brand i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: var(--accent); margin-right: 8px; }
.brand-s { font-size: 13px; color: var(--muted); margin: .25rem 0 .6rem; }
.side-sum { display: grid; gap: 8px; }
.side-sum div { border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; background: var(--canvas); }
.side-sum b { display: block; font-size: 20px; font-weight: 600; }
.side-sum span { font-size: 12px; color: var(--muted); }

/* 로딩바: 앱이 계산하는 동안(화면 전환·질문 처리) 화면 맨 위에 남색 막대가 흐른다 */
.stApp[data-test-script-state="running"]::before { content: ''; position: fixed; top: 0; left: 0; right: 0; height: 3px; z-index: 100001;
  background: linear-gradient(90deg, transparent, var(--accent), var(--sky), transparent) no-repeat; background-size: 35% 100%;
  animation: jdh-bar 1.05s ease-in-out infinite; }
@keyframes jdh-bar { from { background-position: -40% 0; } to { background-position: 140% 0; } }

/* 아이콘 (Material Symbols: 얇은 선 아이콘) */
.ms { font-family: 'Material Symbols Rounded' !important; font-weight: normal; font-style: normal; font-size: 1.25em; line-height: 1;
      letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap; font-feature-settings: 'liga';
      -webkit-font-feature-settings: 'liga'; vertical-align: -0.22em; margin-right: .4em; color: var(--muted);
      font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24; }
.sec .ms, .brand .ms { color: var(--accent); }
.rt .ms { font-size: 1.3em; }
.chip .ms { font-size: 1.15em; margin-right: .25em; vertical-align: -0.2em; color: inherit; }

/* 라벨(Label) */
.chip { display: inline-block; font-size: 12px; font-weight: 500; line-height: 18px; padding: 0 8px; border-radius: 2em; margin-right: 8px; border: 1px solid; }
.chip.e { color: var(--danger); border-color: rgba(255,129,130,.4); background: #ffebe9; }
.chip.w { color: var(--attention); border-color: rgba(212,167,44,.4); background: #fff8c5; }
.chip.i { color: var(--muted); border-color: var(--border); background: var(--subtle); }
.chip.s { color: var(--accent); border-color: #cddaf4; background: #eaf0fa; }
.adv-r { display: flex; gap: 12px; font-size: 14px; line-height: 1.6; margin-top: 8px; }
.adv-l { flex: none; width: 34px; font-weight: 700; color: var(--accent); font-size: 13px; }
.adv-c .adv-t { color: var(--muted); font-size: 13px; }
.adv-wrap { display: grid; grid-template-columns: 150px 1fr; gap: 18px; align-items: start; }
.adv-art { background: var(--tint); border-radius: 8px; padding: 8px; display: flex; align-items: center; justify-content: center; }
.adv-art svg { width: 100%; height: auto; display: block; }
@media (max-width: 860px) { .adv-wrap { grid-template-columns: 1fr; } .adv-art { max-width: 200px; } }

.pill { display: inline-flex; align-items: center; font-size: 12.5px; font-weight: 600; line-height: 22px; padding: 2px 12px 2px 10px; border-radius: 2em; border: 1px solid #dbe3f3; background: #f4f6fb; color: #4a5a7a; }
.pill.ok { background: #f0faf2; border-color: #c3e8cb; color: #1a7f37; }
.pill.ok b { background: #2fbf71; box-shadow: 0 0 0 0 rgba(47,191,113,.55); animation: pill-p 2s ease-out infinite; }
.pill.off b { background: #f5b301; box-shadow: 0 0 0 3px rgba(245,179,1,.2); }
@keyframes pill-p { 0% { box-shadow: 0 0 0 0 rgba(47,191,113,.55); } 70%, 100% { box-shadow: 0 0 0 6px rgba(47,191,113,0); } }
@media (prefers-reduced-motion: reduce) { .pill.ok b { animation: none; } }
.pill-s { font-size: 11.5px; color: var(--muted); margin: 3px 0 0 4px; line-height: 1.35; }
.pill b { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 7px; flex: none; }
</style>
"""


@functools.lru_cache(maxsize=1)
def _mascot_uri() -> str | None:
    """오른쪽 아래에 띄울 캐릭터(character.png). 흰 배경을 지운 투명 PNG 를 assets/ 에 한 번 만들어 둔다."""
    src, out = ROOT / "character.png", ROOT / "assets" / "character_transparent.png"
    try:
        if not out.exists():
            if not src.exists():
                return None
            _make_transparent(src, out)
        return "data:image/png;base64," + base64.b64encode(out.read_bytes()).decode()
    except Exception:
        return None  # 캐릭터가 없거나 처리에 실패해도 서비스는 그대로 동작한다


def _make_transparent(src, out, width: int = 180, thr: int = 250):
    """가장자리와 이어진 순백 배경만 투명하게 만든다(그림 안쪽의 흰색은 그대로 둔다)."""
    from collections import deque

    import numpy as np
    from PIL import Image

    a = np.array(Image.open(src).convert("RGBA"))
    h, w = a.shape[:2]
    white = a[..., :3].min(axis=2) >= thr
    bg = np.zeros((h, w), bool)
    dq = deque()
    for y, x in [(0, i) for i in range(w)] + [(h - 1, i) for i in range(w)] + [(i, 0) for i in range(h)] + [(i, w - 1) for i in range(h)]:
        if white[y, x] and not bg[y, x]:
            bg[y, x] = True
            dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and white[ny, nx] and not bg[ny, nx]:
                bg[ny, nx] = True
                dq.append((ny, nx))
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    near = np.zeros_like(bg)  # 배경과 맞닿은 1픽셀은 반투명으로 부드럽게
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        near |= np.roll(bg, (dy, dx), axis=(0, 1))
    alpha[~bg & near] = 150
    a[..., 3] = alpha
    img = Image.fromarray(a).resize((width, round(h * width / w)), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=True)


def _mascot_style() -> str:
    uri = _mascot_uri()
    if not uri:
        return ""
    # 맨 오른쪽 아래 모서리. 입력창(전송 버튼)과 겹치지 않도록 작게 두고, 화면이 좁으면 숨긴다. 클릭은 그대로 통과(pointer-events:none)
    return ("<style>.stApp::after{content:'';position:fixed;right:10px;bottom:10px;width:84px;height:78px;z-index:100000;"
            f"background:url('{uri}') no-repeat center/contain;opacity:.85;pointer-events:none}}"
            "@media(max-width:1299px){.stApp::after{display:none}}</style>")


def inject_css():
    st.markdown(CSS + _mascot_style(), unsafe_allow_html=True)


def ic(name: str | None) -> str:
    """HTML 안에서 쓰는 선 아이콘. name 은 Material Symbols 이름(예: 'rainy')."""
    return f"<span class='ms'>{name}</span>" if name else ""


from . import maps  # noqa: E402  (지도 HTML 생성은 maps.py)


def show_map(df: pd.DataFrame, height: int = 480, legend: str = ""):
    """지도 표시. 구글 지도 키가 있으면 구글 지도, 없으면 API 키가 필요 없는 OpenStreetMap 지도를 쓴다."""
    from .config import google_maps_key
    if not {"값", "크기", "설명"} <= set(df.columns):
        st.map(df, latitude="lat", longitude="lon", size=120)
        return
    import streamlit.components.v1 as components
    key = google_maps_key()
    doc = maps.google_map_html(df, key, height, legend) if key else maps.leaflet_map_html(df, legend, height)
    if hasattr(st, "iframe"):
        st.iframe(doc, height=height + 8)
    else:  # 구버전 Streamlit
        components.html(doc, height=height + 8)
    st.caption(("구글 지도" if key else "OpenStreetMap") + " · 원을 누르면 상세 정보가 나와요. 크기와 숫자의 뜻은 지도 왼쪽 아래 범례를 보세요.")


def md_safe(text: str) -> str:
    """마크다운에서 '~...~' 가 취소선으로 바뀌지 않게 물결표를 문자 참조로 바꾼다."""
    return text.replace("~", "&#126;")


def split_caveat(text: str) -> tuple[str, str]:
    """답변 본문과 '※ 주의사항'을 분리한다."""
    i = text.find("※")
    return (text[:i].strip(), text[i:].strip()) if i >= 0 else (text.strip(), "")


def note(text: str):
    if text:
        st.markdown(f"<div class='note'>{html.escape(text)}</div>", unsafe_allow_html=True)


def status_pill(ok: bool, text: str, sub: str = ""):
    """AI 연결 상태 표시: 알약 모양 + 작은 보조 문구."""
    st.markdown(f"<span class='pill {'ok' if ok else 'off'}'><b></b>{html.escape(text)}</span>"
                + (f"<div class='pill-s'>{html.escape(sub)}</div>" if sub else ""), unsafe_allow_html=True)


def kpi_cards(items: list[dict]):
    cards = "".join(
        f"<div class='kpi'><div class='l'>{ic(i.get('icon'))}{html.escape(i['label'])}</div>"
        f"<div class='v'>{html.escape(i['value'])}<small>{html.escape(i.get('unit', ''))}</small></div>"
        f"<div class='s'>{html.escape(i.get('sub', ''))}</div></div>" for i in items)
    st.markdown(f"<div class='kpis'>{cards}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ 차트
def _layout(fig: go.Figure, height: int, p: dict):
    fig.update_layout(
        height=height, margin=dict(l=0, r=8, t=8, b=0), showlegend=False, bargap=0.42,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color=p["muted"]), hovermode="closest",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#dbe3f3", font=dict(size=13, color="#1f2a44")),
        xaxis=dict(showgrid=False, linecolor=p["axis"], linewidth=1, ticks="", tickfont=dict(size=12), title=None),
        yaxis=dict(gridcolor=p["grid"], gridwidth=1, griddash="dot", zeroline=False, ticks="", tickfont=dict(size=12), title=None, showline=False),
    )


def bar(df: pd.DataFrame, x: str, y: str, horizontal: bool = False, height: int | None = None, fmt: str = ",.0f",
        dim: list | None = None):
    """막대 차트. 가장 큰 막대만 강조색, 나머지는 연한 단계. 값은 막대 끝에 직접 표기.
    dim 에 든 항목(의미 없는 분류·표본 부족)은 회색으로 빼고 강조 대상에서도 제외한다."""
    p = pal()
    d = df.reset_index(drop=True)
    # 항목 이름이 길고 많으면 가로 막대로 전환(기울어진 글자는 읽기 어렵다)
    if not horizontal and len(d) > 4 and d[x].astype(str).str.len().max() >= 9:
        horizontal = True
    muted = d[x].isin(dim or [])
    vals = d[y].astype(float).where(~muted)
    top = vals.idxmax() if vals.notna().any() else -1
    colors = [p["axis"] if muted[i] else p["accent"] if i == top else p["soft"] for i in range(len(d))]
    kw = dict(marker=dict(color=colors, cornerradius=7), textposition="outside", cliponaxis=False,
              textfont=dict(size=12, color=p["muted"]))
    if horizontal:
        tr = go.Bar(x=d[y], y=d[x], orientation="h", text=d[y], texttemplate=f"%{{text:{fmt}}}",
                    hovertemplate=f"%{{y}}<br>{y}: %{{x:{fmt}}}<extra></extra>", **kw)
    else:
        tr = go.Bar(x=d[x], y=d[y], text=d[y], texttemplate=f"%{{text:{fmt}}}",
                    hovertemplate=f"%{{x}}<br>{y}: %{{y:{fmt}}}<extra></extra>", **kw)
    try:
        fig = go.Figure(tr)
    except ValueError:  # 구버전 plotly: 모서리 둥글게 미지원
        kw["marker"].pop("cornerradius")
        fig = go.Figure(go.Bar(**{**tr.to_plotly_json(), **kw}))
    n = len(d)
    vmax = float(d[y].max()) or 1.0
    _layout(fig, height or (max(220, 34 * n + 30) if horizontal else 300), p)
    if horizontal:
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor=p["grid"], showticklabels=False, linecolor="rgba(0,0,0,0)",
                         range=[0, vmax * 1.2])  # 값 라벨이 잘리지 않도록 여유
    else:
        fig.update_yaxes(showticklabels=True, range=[0, vmax * 1.15])
        if n > 8:
            fig.update_xaxes(tickangle=-30)
    return fig


def line(df: pd.DataFrame, x: str, y: str, height: int = 300):
    p = pal()
    d = df.reset_index(drop=True)
    last = len(d) - 1
    fig = go.Figure(go.Scatter(
        x=d[x], y=d[y], mode="lines", line=dict(color=p["accent"], width=2.4, shape="spline", smoothing=0.6),
        fill="tozeroy", fillcolor=_rgba(p["accent"], 0.09),
        hovertemplate=f"%{{x}}<br>{y}: %{{y:,.0f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[d[x][last]], y=[d[y][last]], mode="markers+text",
                             marker=dict(color=p["accent"], size=9, line=dict(color="#ffffff", width=2.5)),
                             text=[f"{d[y][last]:,.0f}"], textposition="top center", textfont=dict(color=p["muted"], size=12),
                             hoverinfo="skip"))
    _layout(fig, height, p)
    lo, hi = float(d[y].min()), float(d[y].max())
    pad = (hi - lo) * 0.25 or hi * 0.1 or 1.0
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(range=[max(0.0, lo - pad), hi + pad])
    fig.update_xaxes(nticks=8)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def forecast(df: pd.DataFrame, x: str = "월", y: str = "사고 건수", height: int = 320):
    """실적(실선) + 전망(점선) + 80% 범위(음영)."""
    p = pal()
    a, f = df[df["구분"] == "실적"], df[df["구분"] == "전망"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(f[x]) + list(f[x])[::-1], y=list(f["상한"]) + list(f["하한"])[::-1], fill="toself",
                             fillcolor=_rgba(p["accent"], 0.13), line=dict(width=0, shape="spline", smoothing=0.6), hoverinfo="skip", name="80% 범위"))
    fig.add_trace(go.Scatter(x=a[x], y=a[y], mode="lines", line=dict(color=p["accent"], width=2.4, shape="spline", smoothing=0.6), name="실적",
                             hovertemplate=f"%{{x}}<br>실적: %{{y:,.0f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[a[x].iloc[-1]] + list(f[x]), y=[a[y].iloc[-1]] + list(f[y]), mode="lines",
                             line=dict(color=p["accent"], width=2.4, dash="dot", shape="spline", smoothing=0.6), name="전망",
                             hovertemplate=f"%{{x}}<br>전망: %{{y:,.0f}}<extra></extra>"))
    _layout(fig, height, p)
    fig.update_layout(showlegend=True, hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(size=12, color=p["muted"])))
    fig.update_xaxes(nticks=8)
    return fig


def heat(df: pd.DataFrame, x: str, y: str, z: str, height: int | None = None):
    """교차표 히트맵: 칸 안에 건수를 직접 적는다. 색은 한 가지 파랑의 밝기만 바꾼다(글자가 항상 읽히도록 가장 진한 색을 제한)."""
    p = pal()
    rows, cols = list(dict.fromkeys(df[y])), list(dict.fromkeys(df[x]))
    m = df.set_index([y, x])[z].unstack().reindex(index=rows, columns=cols).fillna(0)
    fig = go.Figure(go.Heatmap(
        z=m.values, x=cols, y=rows, colorscale=[[0, "#ffffff"], [1, "#a9bde8"]], showscale=False, xgap=3, ygap=3,
        text=m.values, texttemplate="%{text:,.0f}", textfont=dict(size=13, color="#1f2328"),
        hovertemplate=f"{y}: %{{y}}<br>{x}: %{{x}}<br>{z}: %{{z:,.0f}}<extra></extra>"))
    _layout(fig, height or max(220, 52 * len(rows) + 60), p)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=12))
    fig.update_xaxes(showgrid=False, side="top", tickfont=dict(size=12))
    return fig


# ------------------------------------------------------------------ 움직임 (anime.js)
_MOTION_JS = r"""
(function () {
  if (window.__ajReady) return; window.__ajReady = true;
  if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) return;   // 움직임 줄이기 설정 존중
  var seen = 'data-aj', userMoved = false;
  ['wheel', 'touchstart', 'keydown', 'mousedown'].forEach(function (ev) { window.addEventListener(ev, function () { userMoved = true; }, true); });
  function clear(el) { el.style.opacity = ''; el.style.transform = ''; }
  function fresh(sel) {
    return Array.prototype.filter.call(document.querySelectorAll(sel), function (e) { return !e.hasAttribute(seen); });
  }
  function run(sel, opt) {
    var els = fresh(sel); if (!els.length) return;
    els.forEach(function (e) { e.setAttribute(seen, '1'); e.style.opacity = '0'; });
    anime({ targets: els, opacity: [0, 1], translateY: [opt.y || 12, 0], duration: opt.d || 850, easing: 'easeOutCubic',
            delay: anime.stagger(opt.s || 0, { start: opt.t || 0 }), complete: function () { els.forEach(clear); } });
  }
  function countUp() {
    fresh('.kpi .v').forEach(function (v) {
      v.setAttribute(seen, '1');
      var t = v.firstChild; if (!t || t.nodeType !== 3) return;
      var raw = t.data.trim(), n = parseFloat(raw.replace(/,/g, '')); if (isNaN(n)) return;
      var dec = (raw.split('.')[1] || '').length, o = { x: 0 };
      anime({ targets: o, x: n, duration: 1500, easing: 'easeOutExpo', round: dec ? Math.pow(10, dec) : 1,
              update: function () { t.data = o.x.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }); },
              complete: function () { t.data = raw; } });
    });
  }
  // 차트(plotly): 막대는 바닥에서 자라나고, 선은 왼쪽에서 오른쪽으로 그려진다
  function charts() {
    document.querySelectorAll('.js-plotly-plot .barlayer .trace').forEach(function (tr) {
      var ps = Array.prototype.filter.call(tr.querySelectorAll('.points path'), function (p) { return !p.hasAttribute(seen); });
      if (!ps.length) return;
      ps.forEach(function (p) { p.setAttribute(seen, '1'); });
      var b = ps.map(function (p) { try { return p.getBBox(); } catch (e) { return null; } });
      if (b.some(function (x) { return !x; })) return;
      var vertical = ps.length < 2 || Math.abs((b[0].y + b[0].height) - (b[1].y + b[1].height)) < 1.5;
      ps.forEach(function (p) { p.style.transformBox = 'fill-box'; p.style.transformOrigin = vertical ? '50% 100%' : '0% 50%'; });
      var prop = vertical ? 'scaleY' : 'scaleX', o = { targets: ps, duration: 950, easing: 'easeOutCubic',
        delay: anime.stagger(55), complete: function () { ps.forEach(function (p) { p.style.transform = ''; p.style.transformBox = ''; p.style.transformOrigin = ''; }); } };
      o[prop] = [0, 1]; anime(o);
      var tx = Array.prototype.filter.call(tr.querySelectorAll('.textpoint'), function (t) { return !t.hasAttribute(seen); });
      tx.forEach(function (t) { t.setAttribute(seen, '1'); });
      anime({ targets: tx, opacity: [0, 1], duration: 500, delay: anime.stagger(55, { start: 500 }) });
    });
    fresh('.js-plotly-plot .scatterlayer .trace .lines path.js-line').forEach(function (p) {
      p.setAttribute(seen, '1');
      var dashed = getComputedStyle(p).strokeDasharray;
      if (dashed && dashed !== 'none') { anime({ targets: p, opacity: [0, 1], duration: 900, delay: 700, easing: 'easeOutQuad' }); return; }
      var len = p.getTotalLength ? p.getTotalLength() : 0; if (!len) return;
      p.style.strokeDasharray = len; p.style.strokeDashoffset = len;
      anime({ targets: p, strokeDashoffset: [len, 0], duration: 1300, easing: 'easeInOutSine',
              complete: function () { p.style.strokeDasharray = ''; p.style.strokeDashoffset = ''; } });
    });
    var fills = fresh('.js-plotly-plot .scatterlayer .trace .js-fill');
    fills.forEach(function (f) { f.setAttribute(seen, '1'); });
    if (fills.length) anime({ targets: fills, opacity: [0, 1], duration: 1100, delay: 450, easing: 'easeOutQuad' });
  }
  function scan() {
    charts();
    if (fresh('.hero').length) { userMoved = false; [0, 150, 400, 900, 1700].forEach(function (ms) { setTimeout(function () {
      var mm = document.querySelector('[data-testid="stAppScrollToBottomContainer"]'); if (mm && !userMoved) mm.scrollTo({ top: 0 }); }, ms); }); }
    run('.hero', { y: 18, d: 1050 });
    countUp();
    run('.kpi', { y: 16, s: 120, t: 200, d: 950 });
    run('[class*="st-key-ex"]', { y: 12, s: 85, t: 450, d: 800 });
    run('[data-testid="stChatMessage"]', { y: 14, d: 800 });
    run('.av-card', { y: 16, s: 110, d: 900 });
    run('.sec', { y: 10, d: 750 });
  }
  // 버튼을 누르면 살짝 눌렸다가 탱 하고 튕겨 돌아온다
  var pressed = null;
  function pressable(t) { return t && t.closest ? t.closest('button, [role="radio"], [role="tab"]') : null; }
  window.addEventListener('pointerdown', function (e) {
    var el = pressable(e.target); if (!el || el.disabled) return;
    pressed = el; anime.remove(el);
    anime({ targets: el, scale: 0.92, duration: 110, easing: 'easeOutQuad' });
  }, true);
  function release() {
    var el = pressed; if (!el) return; pressed = null;
    anime.remove(el);
    anime({ targets: el, scale: 1, duration: 750, easing: 'easeOutElastic(1.1, .45)', complete: function () { el.style.transform = ''; } });
  }
  ['pointerup', 'pointercancel', 'dragend'].forEach(function (ev) { window.addEventListener(ev, release, true); });
  var pend = false;
  new MutationObserver(function () { if (pend) return; pend = true; requestAnimationFrame(function () { pend = false; scan(); }); })
    .observe(document.body, { childList: true, subtree: true });
  scan();
})();
"""


_SCROLL_JS = r"""
(function () {
  if (window.__ajScroll) return; window.__ajScroll = true;
  var css = document.createElement('style');
  css.textContent = '#aj-scroll{position:fixed;right:22px;bottom:130px;z-index:100001;display:flex;flex-direction:column;gap:8px}' +
    '#aj-scroll button{width:37px;height:37px;border-radius:50%;border:0;background:#224da0;color:#fff;cursor:pointer;' +
    'box-shadow:0 4px 12px rgba(34,77,160,.35);display:flex;align-items:center;justify-content:center;padding:0;transition:opacity .25s,background .15s}' +
    '#aj-scroll button:hover{background:#00a1e9}#aj-scroll button.off{opacity:0;pointer-events:none}' +
    '#aj-scroll svg{display:block}';
  document.head.appendChild(css);
  var box = document.createElement('div'); box.id = 'aj-scroll';
  function mk(icon, label, fn) {
    var b = document.createElement('button'); b.type = 'button'; b.title = label; b.setAttribute('aria-label', label);
    b.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="' + (icon === 'up' ? '6,15 12,9 18,15' : '6,9 12,15 18,9') + '"/></svg>'; b.onclick = fn; box.appendChild(b); return b;
  }
  function main() { return document.querySelector('[data-testid="stAppScrollToBottomContainer"]') || document.querySelector('[data-testid="stMain"]') || document.querySelector('section.main') || document.scrollingElement; }
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  function go(top) { var m = main(); if (m) m.scrollTo({ top: top < 0 ? m.scrollHeight : top, behavior: reduce ? 'auto' : 'smooth' }); }
  var up = mk('up', '맨 위로', function () { go(0); });
  var down = mk('down', '맨 아래로', function () { go(-1); });
  document.body.appendChild(box);
  function sync() {
    var m = main(); if (!m) return;
    var room = m.scrollHeight - m.clientHeight, y = m.scrollTop;
    up.classList.toggle('off', room < 160 || y < 120);
    down.classList.toggle('off', room < 160 || room - y < 120);
  }
  document.addEventListener('scroll', sync, true); window.addEventListener('resize', sync); setInterval(sync, 500); sync();
})();
"""


def motion():
    """화면 요소가 나타날 때 부드럽게 움직이게 한다(anime.js 3.2.2, MIT — assets/anime.min.js 를 그대로 사용, 인터넷 불필요).
    Streamlit 은 <script> 를 막으므로, 보이지 않는 iframe 이 상위 문서에 스크립트를 한 번만 심는다."""
    import json
    lib = ROOT / "assets" / "anime.min.js"
    if not lib.exists():
        return
    code = (lib.read_text(encoding="utf-8") + ";" + _SCROLL_JS + ";" + _MOTION_JS).replace("</", "<\\/")
    doc = ("<!doctype html><body><script>(function(){try{var P=window.parent,D=P.document;"
           "if(!D.getElementById('aj-lib')){var s=D.createElement('script');s.id='aj-lib';s.text=" + json.dumps(code) + ";D.head.appendChild(s);}"
           "var c=window.frameElement&&window.frameElement.closest('[data-testid=\"stElementContainer\"]');"
           "if(c){c.style.cssText='position:absolute;height:0;width:0;overflow:hidden;margin:0;padding:0'}}catch(e){}})();</script></body>")
    if hasattr(st, "iframe"):
        st.iframe(doc, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(doc, height=1)


# ------------------------------------------------------------------ 로딩 표시
import contextlib


@contextlib.contextmanager
def loading(msg: str, sub: str = ""):
    """분석하는 동안 보여 주는 로딩 표시(신호등 색 점 세 개가 통통 튄다). with 블록이 끝나면 사라진다."""
    ph = st.empty()
    ph.markdown(f"<div class='ldr'><div class='ldr-dots'><i></i><i></i><i></i></div><div><div class='ldr-t'>{html.escape(msg)}</div>"
                + (f"<div class='ldr-s'>{html.escape(sub)}</div>" if sub else "") + "</div></div>", unsafe_allow_html=True)
    try:
        yield
    finally:
        ph.empty()
