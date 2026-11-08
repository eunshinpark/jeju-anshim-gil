"""경로·환경설정. 외부 의존성 없이 .env 를 읽는다."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env():
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()

PAGE_SIZE = 100  # 데이터허브 API 최대 페이지 크기
# 다발지역 API 는 응답에 연도가 없어 제공처 설명의 기간을 문구용으로 둔다(데이터를 바꾸면 환경변수로 수정)
HOTSPOT_PERIOD = os.getenv("TRAFFIC_HOTSPOT_PERIOD", "2017~2023년")


def hotspot_url() -> str | None:
    return os.getenv("TRAFFIC_HOTSPOT_URL")


def google_maps_key() -> str | None:
    """Google Maps JavaScript API 키. 없으면 기본 지도를 쓴다."""
    return os.getenv("GOOGLE_MAPS_API_KEY") or None


def raw_dir() -> Path:
    """원천 CSV 폴더. 기본은 reference/ (없으면 프로젝트 루트). TRAFFIC_RAW_DIR 로 바꿀 수 있다."""
    env = os.getenv("TRAFFIC_RAW_DIR")
    if env:
        return Path(env)
    return ROOT / "reference" if (ROOT / "reference").is_dir() else ROOT


def db_path() -> Path:
    p = Path(os.getenv("TRAFFIC_DB_PATH", "data/jeju_traffic.sqlite"))
    p = p if p.is_absolute() else ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
