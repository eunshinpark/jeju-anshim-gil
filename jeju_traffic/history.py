"""질문 기록: 사용자가 입력한 질문을 이 PC의 파일(data/history.json)에만 저장한다(외부로 보내지 않는다).

화면 왼쪽에 최근 질문을 보여 주고, 누르면 같은 질문을 다시 한다.
"""
from __future__ import annotations

import json
from datetime import datetime

from .config import db_path

MAX_ITEMS = 50


def _file():
    return db_path().parent / "history.json"


def load(limit: int = 20) -> list[dict]:
    """최근 질문 목록(최신순). 파일이 없거나 깨졌으면 빈 목록."""
    try:
        items = json.loads(_file().read_text(encoding="utf-8"))
        return [i for i in items if isinstance(i, dict) and i.get("q")][:limit]
    except Exception:
        return []


def add(q: str) -> None:
    q = (q or "").strip()
    if not q:
        return
    items = [i for i in load(MAX_ITEMS) if i["q"] != q]  # 같은 질문은 맨 위로 올린다
    items.insert(0, {"q": q, "t": datetime.now().isoformat(timespec="seconds")})
    try:
        _file().write_text(json.dumps(items[:MAX_ITEMS], ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass  # 기록 저장에 실패해도 서비스는 계속 동작한다


def clear() -> None:
    try:
        _file().write_text("[]", encoding="utf-8")
    except OSError:
        pass
