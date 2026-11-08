"""Agent: 질문 -> 분석 도구 선택 -> 실행 -> 쉬운 설명.

LLM 경로 (OpenRouter 기본, OpenAI·Anthropic 직접 연결 가능, 선택: 이 PC의 Ollama)
  1) 도구 선택: 모델이 JSON 으로 {"calls":[{"tool","args"}]} 를 답한다 (함수 호출 기능이 없는 모델도 동작).
  2) 설명: 도구 결과(요약문)만 근거로 쉬운 말로 풀어쓴다.
  3) 검증: 설명에 도구 결과에 없는 숫자가 있으면 폐기하고 도구의 요약문을 쓴다. '※' 주의사항이 빠지면 덧붙인다.
폴백: API 키가 없거나(Ollama 는 꺼져 있거나) 호출이 실패하면 키워드 규칙으로 도구를 고르고 요약문을 그대로 사용한다.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import requests

from .analysis import _PED_BY, TOOLS, Result, run_tool, year_span

# 'localhost' 는 Windows 에서 이름 해석에 약 2초가 걸려, 같은 뜻인 127.0.0.1 로 접속한다(화면을 다시 그릴 때마다 묻기 때문에 중요하다)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").replace("//localhost", "//127.0.0.1")

# ---- AI 공급자: openrouter(기본) | openai | anthropic  (외부 서비스, API 키 필요) | ollama(선택·실험: 평가한 gemma3:4b·gpt-oss:20b 는 기준 미달)
PROVIDERS = {
    "openrouter": dict(label="OpenRouter (여러 모델 중 선택)", model="anthropic/claude-haiku-5.5", key_env="OPENROUTER_API_KEY",
                       url="https://openrouter.ai/api/v1/chat/completions"),
    "openai": dict(label="OpenAI (GPT)", model="gpt-4o-mini", key_env="OPENAI_API_KEY",
                   url="https://api.openai.com/v1/chat/completions"),
    "anthropic": dict(label="Anthropic (Claude)", model="claude-haiku-5-5", key_env="ANTHROPIC_API_KEY",
                      url="https://api.anthropic.com/v1/messages"),
    "ollama": dict(label="로컬 Ollama (실험 · 이 PC 안에서 실행)", model="gemma3:4b", key_env=None),
}


def _initial_provider() -> str:
    p = os.getenv("LLM_PROVIDER", "").strip().lower()
    if p in PROVIDERS:
        return p
    for name, cfg in PROVIDERS.items():  # 키가 설정된 외부 서비스가 있으면 그것을 쓴다
        if cfg["key_env"] and os.getenv(cfg["key_env"]):
            return name
    return "openrouter"


PROVIDER = _initial_provider()
API_KEY = os.getenv(PROVIDERS[PROVIDER]["key_env"] or "", "") if PROVIDERS[PROVIDER]["key_env"] else ""
MODEL = os.getenv("LLM_MODEL") or (os.getenv("OLLAMA_MODEL") if PROVIDER == "ollama" else None) or PROVIDERS[PROVIDER]["model"]


def configure(provider: str, model: str | None = None, key: str | None = None) -> None:
    """화면에서 고른 공급자·모델·키를 적용한다(키는 메모리에만 두고 파일에 쓰지 않는다)."""
    global PROVIDER, MODEL, API_KEY
    if provider not in PROVIDERS:
        provider = "openrouter"
    PROVIDER = provider
    MODEL = (model or "").strip() or PROVIDERS[provider]["model"]
    env = PROVIDERS[provider]["key_env"]
    API_KEY = (key or "").strip() or (os.getenv(env, "") if env else "")
    _LLM_CACHE.update(t=0.0, ok=False)
    

SYSTEM = """당신은 제주 교통사고 데이터를 쉽게 설명해 주는 도우미입니다. 사용자는 IT 비전문가인 제주 도민입니다.
규칙:
1. 숫자는 반드시 '분석 결과'에 있는 것만 그대로 사용하고, 직접 계산하거나 새로 만들지 않습니다.
2. 분석 기간을 밝히고, 결과에 '※' 주의사항이 있으면 반드시 전달합니다.
3. 사고 건수가 많은 것과 위험한 것은 다릅니다(교통량 미반영). 원인을 단정하지 말고 '경향'으로 말합니다.
4. 데이터에 없는 질문(보유 기간 밖의 연도, 실시간 정보, 다른 지역, 특정 도로의 교통량)은 없다고 솔직히 말하고 가능한 분석을 제안합니다.
5. 3~5문장 이내의 쉬운 한국어로, 전문용어 없이 설명합니다.
6. 전망 결과는 참고용임을 밝히고, 함께 제시된 검증 오차(평균 오차 %)를 반드시 전달합니다.
7. 분석 결과에 딸린 차트·지도·표는 답변 아래에 자동으로 함께 보입니다. '지도(그림)를 보여드릴 수 없다'고 말하지 말고, 필요하면 '아래 지도에서 확인하세요'라고 안내합니다."""


@dataclass
class Answer:
    text: str
    results: list[Result] = field(default_factory=list)
    mode: str = "rule"  # llm | rule
    notice: str = ""

    @property
    def steps(self) -> list[str]:
        """Agent 가 이어서 부른 분석 도구의 제목(2개 이상일 때만 의미가 있다)."""
        return [r.title for r in self.results] if len(self.results) > 1 else []


def is_external() -> bool:
    """질문과 분석 결과 요약이 이 PC 밖(외부 서비스)으로 나가는 설정인가."""
    return PROVIDER != "ollama"


_LLM_CACHE: dict = {"t": 0.0, "ok": False}


def llm_available(ttl: float = 15.0) -> bool:
    """AI 설명을 쓸 수 있는가. 외부 서비스는 API 키가 있으면 사용 가능으로 본다(화면을 그릴 때마다 인터넷에 묻지 않는다).
    Ollama 는 접속과 모델 설치를 확인하고 ttl 초 동안 결과를 기억한다."""
    if is_external():
        return bool(API_KEY)
    now = time.time()
    if now - _LLM_CACHE["t"] < ttl:
        return _LLM_CACHE["ok"]
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        names = [m["name"] for m in r.json().get("models", [])]
        ok = any(n == MODEL or n.startswith(MODEL.split(":")[0]) for n in names)
    except Exception:
        ok = False
    _LLM_CACHE.update(t=now, ok=ok)
    return ok


# ------------------------------------------------------------ LLM 경로
class LLMError(Exception):
    """AI 서비스 호출 실패. 키는 담지 않고, 원인을 사람이 알아볼 수 있게 요약한다."""


def _check(r) -> None:
    if r.ok:
        return
    try:
        j = r.json()
        msg = j.get("error", j) if isinstance(j, dict) else j
        msg = msg.get("message", msg) if isinstance(msg, dict) else msg
    except Exception:
        msg = r.text
    msg = re.sub(r"\s+", " ", str(msg)).strip()[:160]
    msg = re.sub(r"sk-[\w*.-]{3,}", "sk-***", msg)  # 오류 문구에 키 조각이 섞여 와도 화면에 보이지 않게 한다
    hint = {401: "API 키가 올바르지 않아요", 402: "크레딧이 부족해요", 404: "모델 이름을 확인하세요(또는 저장·학습 금지 조건을 만족하는 제공자가 없는 모델일 수 있어요)",
            429: "요청이 너무 많아요. 잠시 뒤에 다시 해 보세요"}.get(r.status_code, "")
    if r.status_code == 403 and "limit" in msg.lower():
        hint = "API 키의 사용 한도를 넘었어요"
    raise LLMError(f"{r.status_code} {hint or msg}" + (f" ({msg})" if hint and msg else ""))


def _json_text(raw: str) -> str:
    """모델이 ```json 울타리나 앞뒤 설명을 붙여도 JSON 본문만 꺼낸다."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw).strip()
    i, j = raw.find("{"), raw.rfind("}")
    return raw[i:j + 1] if 0 <= i < j else raw


def _openrouter_privacy() -> dict:
    """OpenRouter 로 보낼 때의 개인정보 설정.
    - data_collection=deny : 프롬프트를 저장·학습에 쓸 수 있는 제공자를 제외한다.
    - zdr=true             : 데이터 무보관(Zero Data Retention) 제공자만 쓴다.
    - LLM_PROVIDER_ONLY    : 쓸 제공자를 고정한다(예: azure). 쉼표로 여러 개.
    LLM_PRIVACY=off 로 두면 기본 라우팅을 쓴다(권장하지 않음)."""
    if os.getenv("LLM_PRIVACY", "strict").lower() == "off":
        return {}
    p: dict = {"data_collection": "deny", "zdr": True}
    only = [x.strip() for x in os.getenv("LLM_PROVIDER_ONLY", "").split(",") if x.strip()]
    if only:
        p.update(only=only, allow_fallbacks=False)
    return p


def _chat_external(messages, fmt=None):
    cfg = PROVIDERS[PROVIDER]
    if not API_KEY:
        raise RuntimeError("API 키가 없습니다")
    msgs = [dict(m) for m in messages]
    if fmt:  # 외부 서비스에는 스키마 강제 대신 '이 형식의 JSON 만' 이라고 알려 준다
        hint = "\n\n다음 JSON 스키마에 맞는 JSON 한 개만 출력하세요(설명·코드 울타리 금지):\n" + json.dumps(fmt, ensure_ascii=False)
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] += hint
        else:
            msgs.insert(0, {"role": "system", "content": hint})
    if PROVIDER == "anthropic":
        system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
        body = {"model": MODEL, "max_tokens": 800, "messages": [m for m in msgs if m["role"] != "system"]}
        if system:
            body["system"] = system
        headers = {"x-api-key": API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        r = requests.post(cfg["url"], json=body, headers=headers, timeout=120)
        _check(r)
        return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")
    body = {"model": MODEL, "messages": msgs}
    if PROVIDER == "openai":
        body["max_completion_tokens"] = 1500  # 추론형 모델은 생각에도 토큰을 쓰므로 넉넉히
    else:
        body["max_tokens"] = 800
        body["temperature"] = 0.1
    if fmt:
        body["response_format"] = {"type": "json_object"}
    if PROVIDER == "openrouter":
        body["provider"] = _openrouter_privacy()
    r = requests.post(cfg["url"], json=body, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=120)
    _check(r)
    return r.json()["choices"][0]["message"]["content"] or ""


def _chat(messages, fmt=None):
    if is_external():
        return _chat_external(messages, fmt)
    body = {"model": MODEL, "messages": messages, "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 700}}
    if fmt:
        body["format"] = fmt  # 구조화 출력(JSON 스키마)
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=180)
    _check(r)
    return r.json()["message"]["content"]


def _route_schema():
    return {
        "type": "object",
        "properties": {"calls": {"type": "array", "maxItems": 2, "items": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": list(TOOLS)},
                "args": {"type": "object", "properties": {
                    "factor": {"type": "string", "enum": ["intersection", "alignment", "weather", "surface"]},
                    "target": {"type": "string", "enum": ["elderly", "intersection", "enforcement"]},
                    "by": {"type": "string", "enum": ["hour", "road"]},
                    "city": {"type": "string", "enum": ["제주시", "서귀포시"]},
                    "ped_by": {"type": "string", "enum": list(_PED_BY)}, "rows": {"type": "string", "enum": list(_PED_BY)},
                    "cols": {"type": "string", "enum": list(_PED_BY)}, "victim": {"type": "string", "enum": ["all", "child", "elderly"]},
                    "audience": {"type": "string", "enum": ["all", "citizen", "admin"]},
                    "n": {"type": "integer"}, "horizon": {"type": "integer"},
                    "start_year": {"type": "integer"}, "end_year": {"type": "integer"}}},
            },
            "required": ["tool"]}}},
        "required": ["calls"],
    }


def _route_prompt() -> str:
    lines = ["질문에 답하려면 어떤 분석 도구를 쓸지 고르세요. JSON 으로만 답합니다.", "사용 가능한 도구:"]
    for name, (_, desc, props) in TOOLS.items():
        a = ", ".join(props) if props else "없음"
        lines.append(f"- {name}: {desc} (인자: {a})")
    lines.append("데이터에 없는 질문이거나 도구가 필요 없으면 calls 를 빈 배열로 답합니다. 도구는 최대 2개.")
    return "\n".join(lines)


_NUM = re.compile(r"\d[\d,]*\.?\d*")


def _nums(text: str) -> set[str]:
    return {m.rstrip(".").replace(",", "") for m in _NUM.findall(text)}


_NUM_UNIT = re.compile(r"(\d[\d,]*\.?\d*)\s*([%배건명곳개])?")


def _numbers_grounded(answer: str, source: str) -> bool:
    """설명 속 숫자가 모두 도구 결과에 있는지 검사한다.
    - %·배·건·명·곳·개 같은 단위가 붙은 숫자는 '숫자+단위' 쌍이 근거에 그대로 있어야 한다(1.8배를 1.8%로 바꿔 쓰는 것도 막는다).
    - 단위 없는 숫자는 근거에 있으면 통과하고, 10 이하의 정수는 문장 표현('세 가지' 등)일 수 있어 허용한다."""
    src = _nums(source)
    pairs = {(m.group(1).rstrip(".").replace(",", ""), m.group(2)) for m in _NUM_UNIT.finditer(source) if m.group(2)}
    for m in _NUM_UNIT.finditer(answer):
        n, unit = m.group(1).rstrip(".").replace(",", ""), m.group(2)
        if unit:
            if (n, unit) in pairs:
                continue
            return False
        if n in src or (re.fullmatch(r"\d{1,2}", n) and int(n) <= 10):
            continue
        return False
    return True


def _caveats(summary: str) -> str:
    i = summary.find("※")
    return summary[i:].strip() if i >= 0 else ""


def plan(question: str, history: list[dict] | None = None) -> list[dict]:
    """1단계: 모델이 쓸 도구와 인자를 JSON 으로 고른다. [{'tool':..., 'args':{...}}]"""
    ctx = (history or [])[-2:]
    raw = _chat([{"role": "system", "content": _route_prompt()}, *ctx, {"role": "user", "content": question}],
                fmt=_route_schema())
    return [c for c in json.loads(_json_text(raw)).get("calls", []) if isinstance(c, dict)]


def evidence_of(results: list[Result]) -> str:
    return "\n\n".join(f"[{r.title}] {r.summary}" for r in results)


def explain(question: str, results: list[Result]) -> str:
    """2단계: 도구 결과만 근거로 쉬운 말로 설명한다 (검증 전 원문)."""
    return _chat([{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"질문: {question}\n\n분석 결과:\n{_brief(results)}\n\n위 결과만 근거로 답하세요."}]).strip()


def _brief(results: list[Result]) -> str:
    """모델에게 줄 근거: 요약문에서 '※ 주의사항'은 뺀다(주의사항은 코드가 정확한 원문으로 덧붙인다)."""
    return "\n\n".join(f"[{r.title}] {r.summary.split('※')[0].strip()}" for r in results)


def _clean_args(question: str, args: dict | None) -> dict:
    """AI 가 질문에 없는 연도·개월·개수를 멋대로 채우는 것을 막는다."""
    a = dict(args or {})
    for k in ("start_year", "end_year"):
        if k in a and str(a[k]) not in question:
            a.pop(k)
    if "horizon" in a and not re.search(r"\d+\s*개월|1년|한 ?해|내년", question):
        a.pop("horizon")
    if "n" in a and not re.search(r"\d+\s*(곳|개)", question):
        a.pop("n")
    return a


_CANT_SHOW = re.compile(r"(지도|그림|차트|그래프|표|도표).{0,14}(보여\s*드릴|표시할|제공할|그릴|보여줄|나타낼)\s*수(는|가)?\s*없")


def _trim_incomplete(text: str) -> str:
    """답이 중간에 끊겼으면(문장 부호로 끝나지 않으면) 마지막으로 완성된 문장까지만 남긴다."""
    t = text.rstrip()
    if not t or t[-1] in ".!?。)」\"'":
        return t
    cut = max(t.rfind(c) for c in ".!?。")
    return t[:cut + 1] if cut > 0 else t


def _keep_grounded(text: str, evidence: str) -> tuple[str, int]:
    """문장 단위로 검증한다. 분석 결과에 없는 숫자가 든 문장과 모델이 쓴 '※' 문장은 빼고, 나머지는 살린다."""
    keep, dropped = [], 0
    for s in (x.strip() for x in re.split(r"(?<=[.!?。])\s+|\n+", _trim_incomplete(text)) if x.strip()):
        if s.startswith("※") or _CANT_SHOW.search(s):  # 화면에는 차트·지도가 실제로 나오므로 '못 보여준다'는 문장은 뺀다
            continue
        if _numbers_grounded(s, evidence):
            keep.append(s)
        else:
            dropped += 1
    return " ".join(keep), dropped


def _matched(q: str):
    """키워드로 의도가 분명한 질문이면 도구 목록을, 아니면 None 을 돌려준다."""
    r = _route_chain(q)
    if r == [("data_overview", {})] and not re.search(r"어떤 데이터|무슨 데이터|데이터.*(있|현황|기간)|보유", q):
        return None
    return r


def _ask_llm(question: str, history: list[dict]) -> Answer:
    rule = _matched(question)
    if rule:  # 의도가 분명한 질문은 규칙이 도구를 고른다(빠르고 정확). AI 는 설명을 맡는다.
        calls = [{"tool": n, "args": a} for n, a in rule]
    else:     # 표현이 낯선 질문은 AI 가 도구를 고른다
        calls = [{"tool": c.get("tool", ""), "args": _clean_args(question, c.get("args"))} for c in plan(question, history)]
    results = [run_tool(c["tool"], c.get("args")) for c in calls]
    if any(r.title in ("분석 오류", "오류") for r in results):
        raise ValueError("도구 호출 실패")
    if not results:  # 도구가 필요 없다고 판단 -> 데이터 현황 안내
        results = [run_tool("data_overview", {})]

    evidence = evidence_of(results)
    kept, dropped = _keep_grounded(explain(question, results), evidence)
    notice = ""
    if not kept:
        text = "\n\n".join(r.summary for r in results)
        notice = "AI 설명에 분석 결과에 없는 숫자가 있어 분석 결과 원문으로 대체했습니다."
    else:
        cav = " ".join(c for c in (_caveats(r.summary) for r in results) if c)
        text = kept + ("\n\n" + cav if cav else "")
        if dropped:
            notice = f"AI 설명 중 근거를 확인할 수 없는 문장 {dropped}개를 제외했습니다."
    return Answer(text, results, "llm", notice)


# ------------------------------------------------------------ 규칙 경로 (폴백)
def _years(q: str):
    m = re.search(r"(20\d\d)\s*년?\s*(?:~|-|부터)\s*(20\d\d)", q)
    if m:
        return int(m[1]), int(m[2])
    m = re.search(r"(20\d\d)\s*년", q)
    return (int(m[1]), int(m[1])) if m else (None, None)


_PED_PATTERNS = [
    ("hour", r"시간|몇 시|저녁|밤|새벽|아침|낮|퇴근|출근"),
    ("weekday", r"요일|주말|평일"),
    ("weather", r"비\b|비 오|눈\b|안개|날씨|기상"),
    ("surface", r"노면|젖|결빙"),
    ("violation", r"법규|위반|원인|신호|안전운전"),
    ("victim_age", r"나이|연령|어린이|노인|고령|학생|청소년|세대|어르신"),
    ("road_shape", r"교차로|단일로|도로 ?형태|횡단보도"),
    ("city", r"제주시|서귀포"),
]


def _ped_factors(q: str) -> list[str]:
    """질문에서 언급된 보행자 사고 요인(시간·기상 등)을 찾는다."""
    return [k for k, pat in _PED_PATTERNS if re.search(pat, q)]


_ADVICE = re.compile(r"대안|개선|대책|해결|줄이려|줄일 수|줄이는|예방|조심|주의해야|주의할|안전하게|어떻게 해야|무엇을 해야|뭘 해야|제안|권고")
_NOT_SUBJECT = ("data_overview", "year_notice", "improvement_advice", "quality_report")


def _route_chain(q: str) -> list[tuple[str, dict]]:
    """한 질문에 분석과 후속 요청이 함께 있으면 도구를 이어서 부른다(최대 2단계).
    - '분석 + 대책'  : 먼저 질문 대상을 분석하고, 그 결과에 이어 개선 제안을 붙인다.
    - '추이/어린이 + 어디' : 시기별 분석 뒤에 사고 다발 지점을 이어서 찾는다."""
    base = _route(q)
    if base[0][0] == "improvement_advice":
        rest = _ADVICE.sub(" ", q)
        sub = _route(rest) if len(rest.strip()) > 3 else None
        if sub and sub[0][0] not in _NOT_SUBJECT:
            return [sub[0], base[0]]
        return base
    if base[0][0] in ("child_risk", "monthly_trend") and re.search(r"어디|지도|다발|위험한 곳|위험 지역", q):
        city = "서귀포시" if "서귀포" in q else "제주시" if "제주시" in q else None
        return [base[0], ("hotspot_rank", {"city": city, "n": 10})]
    return base


def _route(q: str) -> list[tuple[str, dict]]:
    s, e = _years(q)
    yr = {"start_year": s, "end_year": e}
    if s is not None:  # 보유 기간 밖의 연도를 물으면 '없다'고 안내한다
        try:
            lo, hi = year_span()
        except Exception:
            lo, hi = 0, 9999
        if s > hi or (e or s) < lo:
            return [("year_notice", {"year": s})]
    if re.search(r"품질|문제점|오류|믿을|신뢰|이상한", q):
        return [("quality_report", {})]
    if _ADVICE.search(q):
        cit = bool(re.search(r"도민|운전자|시민|일반인|부모|조심|주의|안전하게|개인", q))
        adm = bool(re.search(r"담당|기관|행정|도청|정책|시설|예산|점검|개선|대책|지자체|경찰", q))
        return [("improvement_advice", {"audience": "citizen" if cit and not adm else "admin" if adm and not cit else "all"})]
    if re.search(r"예측|전망|예상|앞으로|향후|내년|다음\s?(해|달|년)|미래|얼마나 (될|늘|줄)", q):
        topic = ("enforcement" if re.search(r"단속|음주운전|무면허|과속|안전띠|중앙선", q) else
                 "elderly" if re.search(r"노인|어르신|고령", q) else
                 "other" if re.search(r"어린이|아이|통학|보행자|횡단|비\b|눈\b|안개|날씨|노면|결빙|커브|내리막|교차로|다발|지도|어디", q) else "intersection")
        m = re.search(r"(\d+)\s*개월", q)
        return [("forecast", {"target": topic, "horizon": int(m[1]) if m else 12})]
    if re.search(r"보행자|무단횡단|횡단 ?중|길을 건너", q):
        found = _ped_factors(q)
        if re.search(r"지도|어디|다발|위치|지점|위험한 곳|많이 난 곳", q):
            victim = "child" if re.search(r"어린이|아이|초등", q) else "elderly" if re.search(r"노인|어르신|고령", q) else "all"
            city = "서귀포시" if "서귀포" in q else "제주시" if "제주시" in q else None
            n = re.search(r"(\d+)\s*(곳|개)", q)
            return [("pedestrian_hotspots", {"city": city, "victim": victim, "n": int(n[1]) if n else 15})]
        if len(found) >= 2:
            return [("pedestrian_cross", {"rows": found[0], "cols": found[1]})]
        return [("pedestrian_risk", {"ped_by": found[0] if found else "hour"})]
    if re.search(r"단속|음주운전|무면허|과속|안전띠|중앙선", q):
        return [("enforcement_overview", {})]
    if re.search(r"어떤 데이터|무슨 데이터|데이터.*(있|현황|기간)|보유", q):
        return [("data_overview", {})]
    if re.search(r"어린이|아이|통학|초등", q):
        return [("child_risk", {"by": "road" if "도로" in q else "hour", **yr})]
    if re.search(r"노인|어르신|고령", q):
        return [("monthly_trend", {"target": "elderly", **yr})]
    if re.search(r"다발|위험한 곳|위험 지역|지도|어디|지점|교차로 이름", q):
        city = "서귀포시" if "서귀포" in q else "제주시" if "제주시" in q else None
        n = re.search(r"(\d+)\s*(곳|개)", q)
        return [("hotspot_rank", {"city": city, "n": int(n[1]) if n else 10})]
    if re.search(r"노면|결빙|젖|습기|서리|빙판|미끄", q):
        return [("risk_by_factor", {"factor": "surface", **yr})]
    if re.search(r"비\s?(오|올|가|내)|눈\s?(오|올|이|내)|비\b|눈\b|안개|기상|날씨|맑|흐림|우천|강우", q):
        return [("risk_by_factor", {"factor": "weather", **yr})]
    if re.search(r"커브|곡각|내리막|오르막|직선|도로선형|굽은", q):
        return [("risk_by_factor", {"factor": "alignment", **yr})]
    if re.search(r"교차로|사지|삼지|회전", q):
        return [("risk_by_factor", {"factor": "intersection", **yr})]
    if re.search(r"월별|추이|계절|변화|추세|증가|감소|늘|줄|달은|몇\s?월|(많이|제일).{0,8}달", q):
        return [("monthly_trend", {"target": "intersection", **yr})]
    return [("data_overview", {})]


def _ask_rule(question: str, notice: str = "") -> Answer:
    results = [run_tool(n, a) for n, a in _route_chain(question)]
    return Answer("\n\n".join(r.summary for r in results), results, "rule", notice)


# ------------------------------------------------------------ 진입점
def ask(question: str, history: list[dict] | None = None, use_llm: bool = True) -> Answer:
    if use_llm and llm_available():
        try:
            return _ask_llm(question, history or [])
        except Exception as e:
            why = str(e) if isinstance(e, LLMError) else type(e).__name__
            return _ask_rule(question, f"AI 호출에 실패해({why}) 기본 분석으로 답했습니다.")
    if use_llm and is_external():
        return _ask_rule(question, "API 키가 없어 기본 분석 모드로 답했습니다. 왼쪽 'AI 설정'에서 키를 입력하세요.")
    return _ask_rule(question, "AI 모델(Ollama)이 꺼져 있어 기본 분석 모드로 답했습니다." if use_llm else "")
