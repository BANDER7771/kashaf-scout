"""DeepSeek key management: read from env / .env / Streamlit secrets, or save once from the app into .env."""
from __future__ import annotations

import os

from .config import DEEPSEEK_DEFAULT_MODEL, ROOT

ENV_FILE = ROOT / ".env"


def _read_env_file() -> dict[str, str]:
    if not ENV_FILE.exists():
        return {}
    out = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _secrets() -> dict:
    # Touching st.secrets without a secrets.toml shows an error banner in the app, so check first.
    from pathlib import Path
    if not any(p.exists() for p in (ROOT / ".streamlit" / "secrets.toml", Path.home() / ".streamlit" / "secrets.toml")):
        return {}
    try:
        import streamlit as st
        return dict(st.secrets)
    except Exception:
        return {}


def get(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or _read_env_file().get(name) or _secrets().get(name) or default


def deepseek_key() -> str | None:
    key = get("DEEPSEEK_API_KEY")
    return key if key and not key.endswith("...") else None


def deepseek_model() -> str:
    return get("DEEPSEEK_MODEL", DEEPSEEK_DEFAULT_MODEL)


def save_deepseek_key(key: str) -> None:
    """Persist the key in .env so the app picks it up automatically on every run."""
    values = _read_env_file()
    values["DEEPSEEK_API_KEY"] = key.strip()
    ENV_FILE.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
    os.environ["DEEPSEEK_API_KEY"] = key.strip()


# ---------------------------------------------------------------- hosting (set as secrets on the server)

def hosted() -> bool:
    """On a public server the key comes only from secrets - visitors can never see or change it."""
    return (get("KASHAF_HOSTED") or "").lower() in ("1", "true", "yes")


def app_password() -> str | None:
    return get("APP_PASSWORD") or None


def daily_ai_limit() -> int:
    try:
        return int(get("DAILY_AI_LIMIT") or 0)
    except ValueError:
        return 0


def _usage_file():
    from .config import CACHE_DIR
    import time
    return CACHE_DIR / f"ai_usage_{time.strftime('%Y-%m-%d')}.txt"


def ai_used_today() -> int:
    f = _usage_file()
    try:
        return int(f.read_text()) if f.exists() else 0
    except ValueError:
        return 0


def ai_quota_left() -> int | None:
    """None = unlimited."""
    limit = daily_ai_limit()
    return None if limit <= 0 else max(0, limit - ai_used_today())


def count_ai_use() -> None:
    f = _usage_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(str(ai_used_today() + 1))
