"""수집 실행.

    python -m jeju_traffic.collect                # 폴더의 CSV + API 모두
    python -m jeju_traffic.collect --no-api       # CSV 만
    python -m jeju_traffic.collect --raw D:\\data   # CSV 폴더 지정
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config, quality, sources
from .api import ApiError, hotspots


def _sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12]


def load_csvs(raw: Path):
    tables, meta, skipped = {}, [], []
    for p in sorted(raw.glob("*.csv")):
        if sources.excluded(p):
            continue
        m = sources.match(p)
        if not m:
            skipped.append(p.name)
            continue
        table, fn, desc = m
        df = fn(sources.read_csv(p))
        tables[table] = df
        meta.append(dict(table_name=table, source=p.name, description=desc, rows=len(df), sha1=_sha1(p),
                         loaded_at=datetime.now().isoformat(timespec="seconds")))
    return tables, meta, skipped


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=config.raw_dir())
    ap.add_argument("--no-api", action="store_true")
    a = ap.parse_args(argv)

    tables, meta, skipped = load_csvs(a.raw)
    for s in skipped:
        print(f"  (건너뜀: 처리 방법을 모르는 CSV) {s}")

    if not a.no_api:
        url = config.hotspot_url()
        if not url:
            print("  ! TRAFFIC_HOTSPOT_URL 이 .env 에 없어 다발지역 API는 건너뜁니다.")
        else:
            try:
                tables["hotspots"] = hotspots(url)
                meta.append(dict(table_name="hotspots", source="API: 교통사고 다발지역 정보", description="사고 다발지역",
                                 rows=len(tables["hotspots"]), sha1="", loaded_at=datetime.now().isoformat(timespec="seconds")))
            except ApiError as e:
                print(f"  ! 다발지역 API 실패: {e}")

    tables, issues = quality.check(tables)

    con = sqlite3.connect(config.db_path())
    try:
        for name, df in tables.items():
            df.to_sql(name, con, if_exists="replace", index=False)
        pd.DataFrame(meta).to_sql("source_files", con, if_exists="replace", index=False)
        pd.DataFrame(issues).to_sql("quality_issues", con, if_exists="replace", index=False)
    finally:
        con.close()

    print("\n[저장된 테이블]")
    for m in meta:
        print(f"  {m['table_name']:16s} {m['rows']:>5,}행  {m['description']}  <- {m['source']}")
    print("\n[품질 점검]")
    for sev in ("error", "warn", "info"):
        for i in issues:
            if i["severity"] == sev:
                print(f"  [{sev.upper():5s}] {i['table']}: {i['message']}")
    print(f"\nDB: {config.db_path()}")


if __name__ == "__main__":
    main()
