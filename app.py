"""제주 안심길 - 교통사고 데이터 AI 도우미 (Streamlit). 실행: streamlit run app.py"""
import html

import streamlit as st

from jeju_traffic import advice, agent, diagrams, history, report, ui
from jeju_traffic.analysis import Result, data_overview, kpis, quality_report
from jeju_traffic.config import db_path

st.set_page_config(page_title="제주 안심길", page_icon="🚦", layout="wide")
ui.inject_css()


def _stamp() -> float:
    """DB 파일이 바뀐 시각. 데이터를 다시 수집하면 값이 바뀌어 캐시가 새로 만들어진다."""
    try:
        return db_path().stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def cached_kpis(stamp: float):
    return kpis()

EXAMPLES = [  # (아이콘, 질문, 설명)
    ("location_on", "사고 다발지역을 지도로 보여줘", "반복해서 위험한 곳"),
    ("rainy", "비 오는 날 사고가 더 위험한가요?", "기상과 사고 치명도"),
    ("child_care", "어린이 사고는 몇 시에 많이 나요?", "시간대별 어린이 사고"),
    ("elderly", "노인 교통사고 추이를 보여줘", "월별 추이와 계절성"),
    ("route", "커브길이나 내리막은 얼마나 위험해?", "도로선형과 치명도"),
    ("trending_up", "앞으로 1년 전체 사고를 전망해줘", "월별 전망과 검증 오차"),
    ("query_stats", "노인 교통사고는 앞으로 어떻게 될까?", "전망 범위 포함"),
    ("lightbulb", "교통사고를 줄이려면 어떻게 해야 해?", "개선 제안(도민·담당 기관)"),
    ("tips_and_updates", "도민이 조심해야 할 점을 알려줘", "도민 행동 안내"),
    ("directions_walk", "보행자 사고는 몇 시에 가장 많이 나요?", "보행자 사고 시간대"),
    ("map", "보행자 사고가 많이 난 곳을 지도로 보여줘", "실제 사고 지점"),
    ("grid_view", "연령대별로 보행자 사고가 몇 시에 많은지 보여줘", "연령 × 시간대 교차 분석"),
    ("local_police", "법규위반 단속 현황을 알려줘", "유형별·연도별 단속"),
    ("fact_check", "이 데이터는 믿을 만해?", "자동 품질 점검"),
]
PAGES = [":material/chat_bubble: 질문하기", ":material/description: 자동 점검 보고서", ":material/database: 데이터 현황"]
_TITLE_ICONS = [("개선", "lightbulb"), ("보행자", "directions_walk"), ("단속", "local_police"), ("기상", "rainy"), ("노면", "water_drop"), ("도로선형", "route"), ("교차로", "alt_route"), ("어린이", "child_care"),
                ("노인", "elderly"), ("다발", "location_on"), ("전망", "trending_up"), ("데이터", "database"), ("품질", "fact_check"),
                ("추이", "show_chart")]
_SEC_ICONS = {"1.": "analytics", "2.": "groups", "3.": "directions_walk", "4.": "location_on", "5.": "trending_up", "6.": "lightbulb", "7.": "warning", "부록": "menu_book"}


def _icon_for(title: str):
    return next((ic for kw, ic in _TITLE_ICONS if kw in title), "insights")


def _sec_icon(title: str):
    return next((ic for k, ic in _SEC_ICONS.items() if title.startswith(k)), "article")


def show_chart(spec: dict, df, key: str):
    if spec["kind"] == "heat":
        fig = ui.heat(df, spec["x"], spec["y"], spec["z"])
    elif spec["kind"] == "forecast":
        fig = ui.forecast(df, spec["x"], spec["y"])
    elif spec["kind"] == "line":
        fig = ui.line(df, spec["x"], spec["y"])
    else:
        fig = ui.bar(df, spec["x"], spec["y"], horizontal=bool(spec.get("horizontal")),
                     fmt=",.2f" if spec["y"] in ("치명도", "건당부상") else ",.0f", dim=spec.get("dim"))
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def render(r: Result, key: str):
    with st.container(border=True):
        st.markdown(f"<div class='rt'>{ui.ic(_icon_for(r.title))}{r.title}</div>", unsafe_allow_html=True)
        if r.title.startswith("도로선형별") and not r.df.empty:
            st.markdown(diagrams.alignment_cards(r.df), unsafe_allow_html=True)
        if r.title.startswith("교차로 형태별") and not r.df.empty:
            st.markdown(diagrams.intersection_cards(r.df), unsafe_allow_html=True)
        if r.charts and not r.df.empty:
            cols = st.columns(len(r.charts)) if len(r.charts) > 1 else [st.container()]
            for i, (c, col) in enumerate(zip(r.charts, cols)):
                with col:
                    st.markdown(f"<div class='ct'>{c['y']}</div>", unsafe_allow_html=True)
                    show_chart(c, r.df, f"c{key}_{i}")
        if r.map_df is not None and len(r.map_df):
            ui.show_map(r.map_df, legend=r.map_legend)
        if not r.df.empty:
            with st.expander("표로 보기", expanded=not r.charts and r.map_df is None):
                st.dataframe(r.df, width="stretch", hide_index=True)


# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.markdown(f"<div class='brand'>{ui.ic('traffic')}제주 안심길</div>"
                "<div class='brand-s'>교통사고 데이터 AI 도우미</div>", unsafe_allow_html=True)
    st.session_state.setdefault("page", PAGES[0])
    for _n, _p in enumerate(PAGES):  # 왼쪽 메뉴: 선택된 화면은 진한 남색으로 채운다
        if st.button(_p, key=f"nav_{_n}", type="primary" if st.session_state["page"] == _p else "secondary", width="stretch"):
            st.session_state["page"] = _p
            if _p == PAGES[0]:
                st.session_state["home"] = True  # '질문하기'를 누르면 예시가 있는 시작 화면으로
            st.rerun()
    st.divider()
    use_llm = st.toggle(":material/auto_awesome: AI 해설 사용", value=True, help="AI가 계산된 결과를 쉬운 말로 설명합니다. 꺼져 있거나 모델·키가 없으면 기본 분석 모드로 답합니다.")
    _names = list(agent.PROVIDERS)
    with st.expander("AI 설정", icon=":material/tune:"):
        st.session_state.setdefault("ai_prov", agent.PROVIDER)
        _prov = st.selectbox("AI 서비스", _names, format_func=lambda k: agent.PROVIDERS[k]["label"], key="ai_prov")
        _dflt = agent.MODEL if _prov == agent.PROVIDER else agent.PROVIDERS[_prov]["model"]
        _model = st.text_input("모델 이름", value=_dflt, key=f"ai_model_{_prov}",
                               help="OpenRouter는 'openai/gpt-4o-mini', 'anthropic/claude-haiku-5.5'처럼 '제공자/모델' 형식입니다.")
        _key = ""
        if agent.PROVIDERS[_prov]["key_env"]:
            _key = st.text_input("API 키", type="password", key=f"ai_key_{_prov}", placeholder="환경변수(.env)에 있으면 비워 두세요",
                                 help="이 화면을 여는 동안 메모리에만 보관하고 파일에 저장하지 않습니다. 계속 쓰려면 .env에 넣으세요.")
        agent.configure(_prov, _model, _key)
        if not agent.is_external():
            st.caption(":material/computer: 이 PC 안에서 실행해 데이터가 밖으로 나가지 않아요. Ollama가 켜져 있어야 하고 모델 이름은 `ollama list`로 확인하세요. "
                       "평가에서 gemma3:4b·gpt-oss:20b는 정확도가 기준에 못 미쳤어요(실험 기능).")
        else:
            st.caption(":material/cloud_upload: 질문과 분석 결과 요약이 외부 서비스로 전송됩니다(개인정보는 포함되지 않습니다).")
    ai_on = agent.llm_available()
    ui.status_pill(ai_on, "AI 연결됨" if ai_on else "기본 분석 모드",
                   agent.MODEL if ai_on else "AI 설정에서 키를 넣으면 쉬운 설명이 붙어요" if agent.is_external() else "Ollama를 켜면 쉬운 설명이 붙어요")
    st.divider()
    st.markdown(f"<div class='side-h'>{ui.ic('history')}질문 기록</div>", unsafe_allow_html=True)
    _hist = history.load(15)
    if not _hist:
        st.caption("아직 질문이 없어요. 오른쪽에서 질문해 보세요.")
    for _i, _it in enumerate(_hist):
        _label = _it["q"] if len(_it["q"]) <= 24 else _it["q"][:23] + "…"
        if st.button(_label, key=f"h{_i}", help=_it["q"], icon=":material/history:", width="stretch"):
            st.session_state["pending"] = _it["q"]  # 눌린 질문을 다시 한다
            st.session_state["page"] = PAGES[0]
    if _hist and st.button("기록 지우기", key="hclear", icon=":material/delete:"):
        history.clear()
        st.rerun()

ui.motion()  # 화면이 나타날 때 부드러운 움직임(anime.js)


page = st.session_state["page"]


# ---------------------------------------------------------------- 질문 화면
def chat_page():
    if "log" not in st.session_state:
        st.session_state.log = []
    log = st.session_state.log

    show_home = not log or st.session_state.get("home", False)
    if show_home:
        st.markdown("<div class='hero'><div class='eyebrow'>JEJU TRAFFIC SAFETY</div><h1>제주 교통사고,<br>궁금한 걸 물어보세요</h1>"
                    "<div class='lead'>비 오는 날, 어린이 통학 시간, 커브길까지. 공개 데이터를 분석해 근거와 한계를 함께 알려드려요.</div></div>",
                    unsafe_allow_html=True)
        try:
            ui.kpi_cards(cached_kpis(_stamp()))
        except Exception:
            pass
        st.markdown(f"<div class='sec'>{ui.ic('lightbulb')}이런 걸 물어볼 수 있어요</div>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (icon, q, sub) in enumerate(EXAMPLES):
            with cols[i % 2]:
                if st.button(q, key=f"ex{i}", icon=f":material/{icon}:", width="stretch", help=sub):
                    st.session_state["pending"] = q
        if log and st.button("이전 대화 이어서 보기", key="resume", icon=":material/forum:"):
            st.session_state["home"] = False
            st.rerun()
    for i, turn in enumerate([] if show_home else log):
        with st.chat_message("user", avatar=":material/person:"):
            st.write(turn["q"])
        with st.chat_message("assistant", avatar=":material/traffic:"):
            body, cav = ui.split_caveat(turn["a"].text)
            st.markdown(ui.md_safe(body))
            ui.note(cav)
            if turn["a"].notice:
                st.caption(turn["a"].notice)
            if turn["a"].steps:
                st.caption("분석 단계  " + "  →  ".join(f"{k}. {x}" for k, x in enumerate(turn["a"].steps, 1)))
            for j, r in enumerate(turn["a"].results):
                render(r, f"{i}_{j}")

    q = st.chat_input("질문을 입력하세요  (예: 사고 다발지역을 지도로 보여줘)") or st.session_state.pop("pending", None)
    if q:
        st.session_state["home"] = False
        history.add(q)
        hist = []
        for t in log[-3:]:
            hist += [{"role": "user", "content": t["q"]}, {"role": "assistant", "content": t["a"].text}]
        _ai = use_llm and agent.llm_available()
        with ui.loading("분석하고 있어요", "AI 서비스에 물어보고 있어요" if _ai and agent.is_external() else ""):
            ans = agent.ask(q, hist, use_llm=use_llm)
        log.append({"q": q, "a": ans})
        st.rerun()


# ---------------------------------------------------------------- 보고서 화면
def report_page():
    st.markdown("<div class='hero'><div class='eyebrow'>AUTO REPORT</div><h1>자동 점검 보고서</h1>"
                "<div class='lead'>버튼 한 번으로 전체 데이터를 점검하고, 핵심 발견과 한계를 정리해 드려요.</div></div>", unsafe_allow_html=True)
    ai_ok = use_llm and agent.llm_available()
    rep = st.session_state.get("report") or report.load_latest()  # 이미 만든 보고서가 있으면 바로 보여 준다
    again = rep is not None
    if st.button("보고서 다시 만들기" if again else "보고서 만들기", type="primary",
                 icon=":material/refresh:" if again else ":material/play_arrow:"):
        with ui.loading("전체 데이터를 분석하고 있어요", "보고서를 만드는 중이에요"):
            try:
                rep = report.build(use_llm=ai_ok)
            except FileNotFoundError:
                st.error("DB가 없습니다. `python -m jeju_traffic.collect` 를 먼저 실행하세요.")
                return
        report.save_latest(rep)
    if rep is None:
        try:
            ui.kpi_cards(cached_kpis(_stamp()))
        except Exception:
            pass
        st.info("‘보고서 만들기’를 누르면 분석이 시작됩니다. 완료되면 저장되어 다음에 열 때 바로 보이고, HTML·Markdown 파일로 내려받을 수 있어요.")
        return
    st.session_state["report"] = rep

    stamp = rep.generated_at.replace(":", "").replace("-", "").replace(" ", "_")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("HTML 내려받기", report.to_html(rep).encode("utf-8"), f"제주안심길_점검보고서_{stamp}.html",
                       "text/html", help="인터넷 없이 열립니다. 브라우저 인쇄 → PDF 저장도 가능합니다.", width="stretch", icon=":material/download:")
    c2.download_button("Markdown 내려받기", report.to_markdown(rep).encode("utf-8"), f"제주안심길_점검보고서_{stamp}.md",
                       "text/markdown", width="stretch", icon=":material/download:")
    st.caption(f"작성 {rep.generated_at}" + (" · 데이터가 갱신되었어요. ‘보고서 다시 만들기’로 최신 내용을 반영하세요." if report.latest_is_stale() else ""))
    try:
        ui.kpi_cards(cached_kpis(_stamp()))
    except Exception:
        pass
    if rep.ai_summary:
        with st.container(border=True):
            st.markdown("<div class='rt'>AI 요약</div>", unsafe_allow_html=True)
            st.markdown(ui.md_safe(rep.ai_summary))
    elif ai_ok:
        st.caption("AI 요약은 근거 없는 숫자가 포함되어 생략했습니다. 아래 핵심 발견은 모두 계산된 값입니다.")

    st.markdown(f"<div class='sec'>{ui.ic('insights')}핵심 발견</div>", unsafe_allow_html=True)
    for i, f in enumerate(rep.findings, 1):
        with st.container(border=True):
            st.markdown(f"<div class='find-h'><span class='badge'>{i}</span>{ui.md_safe(f.headline)}</div>"
                        f"<div class='find-e'>근거 · {ui.md_safe(f.evidence)}</div><div class='find-c'>주의 · {ui.md_safe(f.caution)}</div>",
                        unsafe_allow_html=True)

    for si, s in enumerate(rep.sections):
        st.markdown(f"<div class='sec'>{ui.ic(_sec_icon(s.title))}{s.title}</div>", unsafe_allow_html=True)
        for p in s.paragraphs:
            st.markdown(ui.md_safe(p))
        for bi, b in enumerate(s.blocks):
            if b.only == "md":
                continue
            if b.kind == "html":
                with st.container(border=True):
                    st.markdown(f"<div class='rt'>{ui.ic('route')}{b.title}</div>", unsafe_allow_html=True)
                    st.markdown(b.html, unsafe_allow_html=True)
                continue
            if b.df is None:
                continue
            with st.container(border=True):
                st.markdown(f"<div class='rt'>{b.title}</div>", unsafe_allow_html=True)
                if b.kind == "table":
                    st.dataframe(b.df, hide_index=True, width="stretch")
                elif b.kind == "map":
                    ui.show_map(b.df, legend=b.legend)
                else:
                    show_chart({"kind": {"line": "line", "forecast": "forecast"}.get(b.kind, "bar"), "x": b.x, "y": b.y}, b.df, f"rp{si}_{bi}")


# ---------------------------------------------------------------- 개선 제안 화면
def advice_card(a, key: str):
    chip = {"강함": "s", "보통": "w", "약함": "i"}[a.strength]
    esc = lambda t: ui.md_safe(html.escape(t))
    with st.container(border=True):
        body = (f"<div class='find-h'><span class='chip {chip}'>근거 {a.strength}</span>{esc(a.title)}</div>"
                f"<div class='adv-r'><span class='adv-l'>근거</span><span class='adv-t'>{esc(a.basis)}</span></div>"
                f"<div class='adv-r'><span class='adv-l'>제안</span><span class='adv-t'>{esc(a.action)}</span></div>"
                f"<div class='adv-r adv-c'><span class='adv-l'>한계</span><span class='adv-t'>{esc(a.caveat)}</span></div>")
        art = diagrams.advice_art(a.art)
        st.markdown(f"<div class='adv-wrap'><div class='adv-art'>{art}</div><div>{body}</div></div>" if art else body,
                    unsafe_allow_html=True)
        if a.items:
            st.caption("우선 점검 지점 · " + " / ".join(a.items))
        if a.map_df is not None:
            with st.expander("지도에서 보기"):
                ui.show_map(a.map_df, legend=a.map_legend)


# ---------------------------------------------------------------- 데이터 현황 화면
def data_page():
    st.markdown("<div class='hero'><div class='eyebrow'>DATA</div><h1>데이터 현황</h1>"
                "<div class='lead'>어떤 데이터를 쓰고 있는지, 믿어도 되는지 자동으로 점검한 결과예요.</div></div>", unsafe_allow_html=True)
    try:
        ov, q = data_overview().df, quality_report().df
    except Exception:
        st.error("DB가 없습니다. `python -m jeju_traffic.collect` 를 먼저 실행하세요.")
        return

    st.markdown(f"<div class='sec'>{ui.ic('database')}보유 데이터</div>", unsafe_allow_html=True)
    st.dataframe(ov[["설명", "행수", "기간", "출처파일"]], hide_index=True, width="stretch",
                 column_config={"기간": st.column_config.TextColumn(width="large"),
                                "출처파일": st.column_config.TextColumn(width="large")})

    name = dict(zip(ov.테이블, ov.설명))
    kinds = {"error": ("오류", "e", "error"), "warn": ("주의", "w", "warning"), "info": ("참고", "i", "info")}
    n = {k: int((q.등급 == k).sum()) for k in kinds}
    st.markdown(f"<div class='sec'>{ui.ic('fact_check')}품질 점검</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='lead' style='margin-bottom:.8rem'>오류 <b>{n['error']}</b>건 · 주의 <b>{n['warn']}</b>건 · "
                f"정상 확인/참고 {n['info']}건. 분석에는 아래 조치가 이미 반영돼 있어요.</div>", unsafe_allow_html=True)
    for r in q[q.등급.isin(["error", "warn"])].itertuples():
        label, cls, icn = kinds[r.등급]
        action = report.ACTIONS.get(r.코드)
        with st.container(border=True):
            st.markdown(f"<span class='chip {cls}'>{ui.ic(icn)}{label}</span><b>{name.get(r.테이블, r.테이블)}</b>", unsafe_allow_html=True)
            st.markdown(ui.md_safe(r.내용))
            if action:
                st.markdown(f"<div class='note'>반영한 조치 · {ui.md_safe(action)}</div>", unsafe_allow_html=True)
    info = q[q.등급 == "info"].copy()
    info["테이블"] = info.테이블.map(lambda t: name.get(t, t))
    with st.expander(f"정상 확인·참고 사항 ({len(info)}건)"):
        st.dataframe(info[["테이블", "내용"]], hide_index=True, width="stretch")


if page == PAGES[1]:
    report_page()
elif page == PAGES[2]:
    data_page()
else:
    chat_page()
