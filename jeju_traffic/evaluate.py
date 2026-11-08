"""AI 모델 검증. 사용할 모델이 충분한지 숫자로 판단한다.

    python -m jeju_traffic.evaluate                          # 기본 모델(gemma3:4b)
    python -m jeju_traffic.evaluate --models gemma3:4b,llama3.1:8b   # 여러 모델 비교
    python -m jeju_traffic.evaluate --limit 8 --no-explain   # 빠른 점검(도구 선택만)

측정 항목
  1. 연결      : API 키·서비스 접속
  2. 도구 선택 : 질문에 맞는 분석 도구·핵심 인자를 골랐는지 (키워드 규칙과 비교)
  3. 설명 충실도: 설명 속 숫자가 모두 분석 결과에 있는지(지어낸 숫자 없음), ※ 주의사항 포함, 한국어 비율
  4. 범위 밖 질문: 데이터에 없는 질문(먼 미래 연도, 실시간, 타 지역)에 '없다'고 답하는지
  5. 속도·자원  : 단계별 지연 시간(중앙값/p95), GPU 적재 비율(ollama ps)
※ 판정은 아래 기준표를 따르며, 모델이 틀려도 규칙 폴백·숫자 검증이 있어 서비스는 계속 동작한다.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
import time
from datetime import datetime

import requests

from . import agent
from .analysis import run_tool
from .config import ROOT

# in: 정답 도구(집합)와 핵심 인자 / out: 데이터 범위 밖 -> 도구 없음 또는 data_overview 허용
CASES = [
    dict(q="비 오는 날 사고가 더 위험한가요?", tools={"risk_by_factor"}, args={"factor": "weather"}),
    dict(q="눈이나 안개 낀 날은 어때요?", tools={"risk_by_factor"}, args={"factor": "weather"}),
    dict(q="비올 때 교통사고 얼마나 나?", tools={"risk_by_factor"}, args={"factor": "weather"}),
    dict(q="길이 젖어 있으면 사고가 더 나나요?", tools={"risk_by_factor"}, args={"factor": "surface"}),
    dict(q="빙판길 사고 현황 알려줘", tools={"risk_by_factor"}, args={"factor": "surface"}),
    dict(q="커브길은 얼마나 위험해?", tools={"risk_by_factor"}, args={"factor": "alignment"}),
    dict(q="내리막길 사고 치명도가 궁금해요", tools={"risk_by_factor"}, args={"factor": "alignment"}),
    dict(q="직선도로랑 커브길을 비교해줘", tools={"risk_by_factor"}, args={"factor": "alignment"}),
    dict(q="교차로 형태별로 사고를 비교해줘", tools={"risk_by_factor"}, args={"factor": "intersection"}),
    dict(q="회전교차로는 안전한 편인가요?", tools={"risk_by_factor"}, args={"factor": "intersection"}),
    dict(q="사지 교차로에서 사고가 많이 나나요?", tools={"risk_by_factor"}, args={"factor": "intersection"}),
    dict(q="어린이 사고는 몇 시에 많이 나요?", tools={"child_risk"}, args={"by": "hour"}),
    dict(q="아이들 하교 시간에 사고가 많은가요?", tools={"child_risk"}, args={"by": "hour"}),
    dict(q="어린이 사고가 많은 도로 종류는 뭐예요?", tools={"child_risk"}, args={"by": "road"}),
    dict(q="노인 교통사고 추이를 보여줘", tools={"monthly_trend"}, args={"target": "elderly"}),
    dict(q="어르신 교통사고가 늘고 있나요?", tools={"monthly_trend"}, args={"target": "elderly"}),
    dict(q="전체 교통사고 월별 추이를 알려줘", tools={"monthly_trend"}, args={"target": "intersection"}),
    dict(q="사고가 제일 많이 나는 달은 언제야?", tools={"monthly_trend"}, args={"target": "intersection"}),
    dict(q="서귀포 사고 다발지역 3곳 알려줘", tools={"hotspot_rank"}, args={"city": "서귀포시"}),
    dict(q="제주시에서 위험한 곳이 어디야?", tools={"hotspot_rank"}, args={"city": "제주시"}),
    dict(q="사고 다발지역을 지도로 보여줘", tools={"hotspot_rank"}, args={}),
    dict(q="이 데이터는 믿을 만해?", tools={"quality_report"}, args={}),
    dict(q="어떤 데이터를 볼 수 있어?", tools={"data_overview"}, args={}),
    dict(q="2019년 교차로 사고를 알려줘", tools={"risk_by_factor"}, args={"factor": "intersection", "start_year": 2019}),
    dict(q="앞으로 1년 전체 사고를 전망해줘", tools={"forecast"}, args={"target": "intersection"}),
    dict(q="노인 교통사고는 앞으로 어떻게 될까?", tools={"forecast"}, args={"target": "elderly"}),
    dict(q="다음 6개월 사고 건수를 예측해줘", tools={"forecast"}, args={"target": "intersection", "horizon": 6}),
    dict(q="보행자 사고는 몇 시에 가장 많이 나요?", tools={"pedestrian_risk"}, args={"ped_by": "hour"}),
    dict(q="비 오는 날 저녁 보행자 사고는 어때?", tools={"pedestrian_cross"}, args={}),
    dict(q="보행자 사고가 많이 난 곳을 지도로 보여줘", tools={"pedestrian_hotspots"}, args={}),
    dict(q="어린이 보행자 사고가 많은 지점은?", tools={"pedestrian_hotspots"}, args={"victim": "child"}),
    dict(q="법규위반 단속 현황을 알려줘", tools={"enforcement_overview"}, args={}),
    dict(q="음주운전 단속 건수는 앞으로 어떻게 될까?", tools={"forecast"}, args={"target": "enforcement"}),
    dict(q="교통사고를 줄이려면 어떻게 해야 해?", tools={"improvement_advice"}, args={}),
    dict(q="도민이 조심해야 할 점을 알려줘", tools={"improvement_advice"}, args={"audience": "citizen"}),
    dict(q="개선이 필요한 곳은 어디야?", tools={"improvement_advice"}, args={"audience": "admin"}),
    # 데이터 범위 밖
    dict(q="2040년 교통사고는 몇 건이야?", out=True),
    dict(q="지금 제주시 도로 상황 알려줘", out=True),
    dict(q="서울 교통사고 현황은 어때?", out=True),
    dict(q="오늘 점심 뭐 먹을까?", out=True),
]

# 판정 기준: (도구 선택, 인자, 지어낸 숫자 없음, JSON 성공, p95 총 지연 초)
GOOD = dict(tool=0.85, args=0.80, grounded=0.95, parse=0.95, p95=45)
OK = dict(tool=0.70, args=0.60, grounded=0.85, parse=0.85, p95=90)
_NEG = re.compile(r"없|않|못|제공|확인할 수|지원|범위|알 수")


def _hangul_ratio(t: str) -> float:
    letters = re.findall(r"[가-힣A-Za-z]", t)
    return len(re.findall(r"[가-힣]", t)) / len(letters) if letters else 0.0


def _sentences(t: str) -> int:
    return len([s for s in re.split(r"[.!?。]\s*|\n+", t) if s.strip()])


def _plan_ok(case: dict, calls: list[dict]) -> tuple[bool, bool]:
    tools = {c.get("tool") for c in calls}
    if case.get("out"):
        ok = tools in (set(), {"data_overview"}, {"year_notice"})
        return ok, ok
    ok_tool = tools == case["tools"]
    if not ok_tool:
        return False, False
    merged = {}
    for c in calls:
        merged.update({k: v for k, v in (c.get("args") or {}).items() if v not in (None, "")})
    ok_args = all(str(merged.get(k)) == str(v) for k, v in case["args"].items())
    return ok_tool, ok_args


def _rule_calls(q: str) -> list[dict]:
    return [{"tool": n, "args": a} for n, a in agent._route(q)]


def preflight(url: str, model: str) -> tuple[bool, str]:
    if agent.is_external():
        return (True, "OK") if agent.API_KEY else (False, f"API 키가 없습니다. 환경변수 {agent.PROVIDERS[agent.PROVIDER]['key_env']} 를 설정하세요.")
    try:
        r = requests.get(f"{url}/api/tags", timeout=3)
        r.raise_for_status()
    except Exception as e:
        return False, (f"Ollama 에 접속할 수 없습니다({type(e).__name__}).\n"
                       f"  1) Ollama 설치/실행 확인  2) 주소 확인(현재 {url}, 환경변수 OLLAMA_URL)")
    names = [m["name"] for m in r.json().get("models", [])]
    if not (model in names or (":" not in model and f"{model}:latest" in names)):
        return False, f"모델 '{model}' 이(가) 설치되어 있지 않습니다.\n  설치: ollama pull {model}\n  설치된 모델: {', '.join(names) or '없음'}"
    return True, "OK"


def gpu_share(url: str, model: str):
    if agent.is_external():
        return None, 0
    try:
        for m in requests.get(f"{url}/api/ps", timeout=3).json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return (m.get("size_vram", 0) / m["size"]) if m.get("size") else None, m.get("size", 0)
    except Exception:
        pass
    return None, 0


def _pct(x, n):
    return f"{100 * x / n:.0f}%" if n else "-"


def _p95(v):
    return sorted(v)[max(int(round(0.95 * len(v))) - 1, 0)] if v else 0.0


def run_model(model: str, cases, do_explain: bool, verbose: bool = True) -> dict:
    agent.MODEL = model
    ok, msg = preflight(agent.OLLAMA_URL, model)
    if not ok:
        return dict(model=model, error=msg)

    t0 = time.time()
    try:
        agent._chat([{"role": "user", "content": "안녕"}])  # 모델 적재(워밍업) - 지연 통계에서 제외
    except Exception as e:
        return dict(model=model, error=f"워밍업 실패({type(e).__name__}: {e})")
    load_s = time.time() - t0

    rows = []
    for i, c in enumerate(cases, 1):
        row = dict(q=c["q"], out=bool(c.get("out")), parse_ok=True, tool_ok=False, args_ok=False,
                   rule_tool_ok=False, rule_args_ok=False)
        rc = _rule_calls(c["q"]) if not c.get("out") else []
        row["rule_tool_ok"], row["rule_args_ok"] = _plan_ok(c, rc) if not c.get("out") else (False, False)
        t1 = time.time()
        try:
            calls = agent.plan(c["q"])
        except Exception as e:
            calls, row["parse_ok"], row["err"] = [], False, f"{type(e).__name__}"
        row["plan_s"] = time.time() - t1
        row["calls"] = [{"tool": x.get("tool"), "args": x.get("args")} for x in calls]
        row["tool_ok"], row["args_ok"] = _plan_ok(c, calls) if row["parse_ok"] else (False, False)

        if do_explain and row["parse_ok"]:
            results = [run_tool(x.get("tool", ""), x.get("args")) for x in calls] or [run_tool("data_overview", {})]
            ev = agent.evidence_of(results)
            t2 = time.time()
            try:
                text = agent.explain(c["q"], results)
                row["explain_s"] = time.time() - t2
                row["text"] = text
                row["grounded"] = agent._numbers_grounded(text, ev)
                row["has_caveat"] = ("※" in text) if any("※" in r.summary for r in results) else None
                row["hangul"] = round(_hangul_ratio(text), 2)
                row["sentences"] = _sentences(text)
                row["honest"] = bool(_NEG.search(text)) if c.get("out") else None
            except Exception as e:
                row["explain_err"] = type(e).__name__
        rows.append(row)
        if verbose:
            mark = "✓" if row["tool_ok"] and row["args_ok"] else "✗"
            print(f"  [{i:>2}/{len(cases)}] {mark} {c['q'][:30]:30s} -> {[x['tool'] for x in row['calls']] or '도구 없음'}"
                  f"  ({row['plan_s']:.1f}s)", flush=True)

    gpu, size = gpu_share(agent.OLLAMA_URL, model)
    inn, out = [r for r in rows if not r["out"]], [r for r in rows if r["out"]]
    exp = [r for r in rows if "grounded" in r]
    tot = [r["plan_s"] + r.get("explain_s", 0) for r in rows]
    s = dict(
        model=model, n=len(rows), n_in=len(inn), n_out=len(out), load_s=round(load_s, 1),
        parse=sum(r["parse_ok"] for r in rows) / len(rows),
        tool=sum(r["tool_ok"] for r in inn) / len(inn) if inn else None,
        args=sum(r["args_ok"] for r in inn) / len(inn) if inn else None,
        rule_tool=sum(r["rule_tool_ok"] for r in inn) / len(inn) if inn else None,
        rule_args=sum(r["rule_args_ok"] for r in inn) / len(inn) if inn else None,
        out_ok=sum(r["tool_ok"] for r in out) / len(out) if out else None,
        grounded=sum(r["grounded"] for r in exp) / len(exp) if exp else None,
        caveat=(lambda v: sum(v) / len(v) if v else None)([r["has_caveat"] for r in exp if r.get("has_caveat") is not None]),
        hangul=st.mean(r["hangul"] for r in exp) if exp else None,
        long=sum(r["sentences"] > 6 for r in exp) / len(exp) if exp else None,
        honest=(lambda v: sum(v) / len(v) if v else None)([r["honest"] for r in exp if r.get("honest") is not None]),
        plan_med=st.median(r["plan_s"] for r in rows), plan_p95=_p95([r["plan_s"] for r in rows]),
        explain_med=st.median([r["explain_s"] for r in exp]) if exp else None,
        total_med=st.median(tot), total_p95=_p95(tot),
        gpu=gpu, size_gb=round(size / 1e9, 1) if size else None,
        rows=rows,
    )
    s["verdict"], s["reasons"] = verdict(s)
    return s


def verdict(s: dict) -> tuple[str, list[str]]:
    def val(k):
        return s.get(k)
    checks = [("도구 선택", "tool", ">="), ("핵심 인자", "args", ">="), ("지어낸 숫자 없음", "grounded", ">="), ("JSON 성공", "parse", ">=")]
    reasons, level = [], 2  # 2=사용 가능, 1=조건부, 0=부적합
    for name, k, _ in checks:
        v = val(k)
        if v is None:
            continue
        if v < OK[k]:
            level = 0
            reasons.append(f"{name} {v:.0%} (최소 {OK[k]:.0%} 미달)")
        elif v < GOOD[k]:
            level = min(level, 1)
            reasons.append(f"{name} {v:.0%} (권장 {GOOD[k]:.0%} 미달)")
    p95 = val("total_p95")
    if p95 > OK["p95"]:
        level = 0
        reasons.append(f"느림: p95 {p95:.0f}초 (한도 {OK['p95']}초)")
    elif p95 > GOOD["p95"]:
        level = min(level, 1)
        reasons.append(f"다소 느림: p95 {p95:.0f}초 (권장 {GOOD['p95']}초)")
    g = val("gpu")
    if g is not None and g < 0.9:
        level = min(level, 1)
        reasons.append(f"GPU 적재 {g:.0%} — 일부가 CPU 로 처리되어 느릴 수 있음(더 작은 모델/양자화 고려)")
    return ["부적합", "조건부", "사용 가능"][level], reasons


def _f(x, pct=True):
    return "-" if x is None else (f"{x:.0%}" if pct else f"{x:.1f}")


def markdown(results: list[dict]) -> str:
    where = agent.PROVIDERS[agent.PROVIDER]["label"] + ("" if agent.is_external() else f" {agent.OLLAMA_URL}")
    o = [f"# AI 모델 검증 결과\n", f"실행: {datetime.now():%Y-%m-%d %H:%M} · {where}\n"]
    ok = [r for r in results if "error" not in r]
    for r in results:
        if "error" in r:
            o.append(f"- **{r['model']}**: 실행하지 못함 — {r['error']}\n")
    if ok:
        o.append("| 모델 | 판정 | 도구 선택 | (규칙) | 핵심 인자 | JSON | 지어낸 숫자 없음 | ※ 포함 | 범위 밖 | 총 지연 중앙/p95 | GPU |")
        o.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in ok:
            o.append(f"| {r['model']} | **{r['verdict']}** | {_f(r['tool'])} | {_f(r['rule_tool'])} | {_f(r['args'])} | {_f(r['parse'])} | "
                     f"{_f(r['grounded'])} | {_f(r['caveat'])} | {_f(r['out_ok'])} | {r['total_med']:.1f}s / {r['total_p95']:.1f}s | "
                     f"{_f(r['gpu'])} |")
        for r in ok:
            o.append(f"\n## {r['model']} — {r['verdict']}")
            o.append("- 사유: " + ("; ".join(r["reasons"]) if r["reasons"] else "모든 기준 충족"))
            o.append(f"- 모델 적재 {r['load_s']}초, 크기 {r['size_gb']}GB, 한국어 비율 {_f(r['hangul'])}, "
                     f"6문장 초과 {_f(r['long'])}, 범위 밖 질문에서 '없다'고 답함 {_f(r['honest'])}")
            bad = [x for x in r["rows"] if not (x["tool_ok"] and x["args_ok"])]
            if bad:
                o.append("\n틀린 질문:\n")
                for x in bad:
                    got = [c["tool"] for c in x["calls"]] or "도구 없음"
                    if x["out"]:
                        o.append(f"- “{x['q']}” → {got} (범위 밖 질문이므로 도구를 쓰지 않아야 함)")
                    elif x["tool_ok"]:  # 도구는 맞고 인자만 틀림
                        case = next(c for c in CASES if c["q"] == x["q"])
                        have = {k: v for cc in x["calls"] for k, v in (cc.get("args") or {}).items()}
                        o.append(f"- “{x['q']}” → 도구는 맞았지만 인자가 다름 (기대 {case['args']}, 실제 {have})")
                    else:
                        case = next(c for c in CASES if c["q"] == x["q"])
                        o.append(f"- “{x['q']}” → {got} (기대: {sorted(case['tools'])})" + (f" [{x['err']} - JSON 해석 실패]" if x.get("err") else ""))
            fab = [x for x in r["rows"] if x.get("grounded") is False]
            if fab:
                o.append("\n근거 없는 숫자가 있던 설명:\n")
                for x in fab:
                    o.append(f"- “{x['q']}”: {x['text'][:140]}")
    o.append("\n## 판정 기준\n")
    o.append(f"- 사용 가능: 도구 선택 ≥ {GOOD['tool']:.0%}, 핵심 인자 ≥ {GOOD['args']:.0%}, 지어낸 숫자 없음 ≥ {GOOD['grounded']:.0%}, JSON ≥ {GOOD['parse']:.0%}, p95 ≤ {GOOD['p95']}초")
    o.append(f"- 조건부: 도구 ≥ {OK['tool']:.0%}, 인자 ≥ {OK['args']:.0%}, 숫자 ≥ {OK['grounded']:.0%}, JSON ≥ {OK['parse']:.0%}, p95 ≤ {OK['p95']}초 (규칙 폴백·숫자 검증으로 서비스는 동작)")
    o.append("- 그 이하: 부적합 → 더 큰/다른 모델을 시도하거나 규칙 모드로 운영")
    o.append("- '(규칙)' 열은 AI 없이 키워드 규칙만 썼을 때의 도구 선택 정확도입니다. AI가 규칙보다 낮으면 AI 연결의 이점이 없다는 뜻입니다. "
             "단, 규칙은 이 점검 질문으로 다듬었기 때문에 실제 사용자 질문에서는 이보다 낮을 수 있습니다(AI는 처음 보는 표현에 강점).")
    return "\n".join(o)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="쉼표로 구분한 모델 목록(기본: 선택한 AI 서비스의 기본 모델)")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개 질문만(빠른 점검). 범위 밖 질문도 일부 포함")
    ap.add_argument("--provider", default=None, choices=list(agent.PROVIDERS), help="AI 서비스(기본: 환경변수/openrouter). 외부 서비스는 API 키 환경변수 필요")
    ap.add_argument("--no-explain", action="store_true", help="도구 선택만 점검(빠름)")
    a = ap.parse_args(argv)
    if a.provider:
        agent.configure(a.provider)

    cases = CASES
    if a.limit:
        ins = [c for c in CASES if not c.get("out")]
        outs = [c for c in CASES if c.get("out")]
        cases = ins[: max(a.limit - 2, 1)] + outs[:2]

    results = []
    for m in [x.strip() for x in (a.models or agent.MODEL).split(",") if x.strip()]:
        print(f"\n=== {m} ({len(cases)}문항) ===", flush=True)
        r = run_model(m, cases, do_explain=not a.no_explain)
        if "error" in r:
            print("  ✗ " + r["error"])
        else:
            print(f"  → 판정: {r['verdict']} | 도구 {_f(r['tool'])} 인자 {_f(r['args'])} 숫자충실 {_f(r['grounded'])} "
                  f"지연(총) 중앙 {r['total_med']:.1f}s p95 {r['total_p95']:.1f}s GPU {_f(r['gpu'])}")
        results.append(r)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    md = out / f"모델검증_{stamp}.md"
    md.write_text(markdown(results), encoding="utf-8")
    (out / f"모델검증_{stamp}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n결과 저장: {md}")
    return 0 if any("error" not in r for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
