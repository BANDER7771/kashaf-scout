"""Understat (xG data for top-5 leagues + RFPL) with a small on-disk cache."""
from __future__ import annotations

import json
import time

import pandas as pd
import requests
from rapidfuzz import fuzz

from .config import CACHE_DIR, UNDERSTAT_URL
from .data_tm import normalize_name

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
}
NUMERIC = ["games", "time", "goals", "xG", "assists", "xA", "shots", "key_passes", "npg", "npxG",
           "xGChain", "xGBuildup", "yellow_cards", "red_cards"]


def _get_json(path: str, cache_key: str, ttl_hours: float) -> dict | None:
    cache = CACHE_DIR / "understat" / f"{cache_key}.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < ttl_hours * 3600:
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        r = requests.get(UNDERSTAT_URL + path, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        # Fall back to a stale cache rather than failing the whole profile.
        return json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def league_players(league: str, season: int) -> pd.DataFrame:
    """Season totals for every player in a league. season = start year (2025 -> 2025/26)."""
    current = season >= pd.Timestamp.now().year - 1
    data = _get_json(f"/getLeagueData/{league}/{season}", f"league_{league}_{season}",
                     ttl_hours=24 if current else 24 * 365)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data["players"])
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["league"] = league
    df["season"] = season
    return df


def player_data(understat_id: str) -> dict | None:
    return _get_json(f"/getPlayerData/{understat_id}", f"player_{understat_id}", ttl_hours=24)


def find_player(name: str, league: str, season: int, club_hint: str | None = None) -> str | None:
    """Fuzzy-match a Transfermarkt player to an Understat id within one league-season."""
    df = league_players(league, season)
    if df.empty:
        return None
    target = normalize_name(name)
    df = df.assign(name_score=df["player_name"].map(lambda n: fuzz.WRatio(target, normalize_name(n))))
    if club_hint:
        hint = normalize_name(club_hint)
        df["name_score"] += df["team_title"].map(lambda t: 8 if fuzz.partial_ratio(hint, normalize_name(t)) > 75 else 0)
    best = df.sort_values("name_score", ascending=False).iloc[0]
    return str(best["id"]) if best["name_score"] >= 85 else None


def player_matches(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data.get("matches", []))
    if df.empty:
        return df
    for c in ["goals", "shots", "xG", "time", "xA", "assists", "key_passes", "npg", "npxG", "xGChain", "xGBuildup"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = pd.to_numeric(df["season"])
    return df.sort_values("date")


def player_seasons(data: dict) -> pd.DataFrame:
    """One row per season (a mid-season move is split per team on Understat - summed here)."""
    df = pd.DataFrame(data.get("groups", {}).get("season", []))
    if df.empty:
        return df
    for c in df.columns:
        if c not in ("position", "team"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    num = df.drop(columns=["position", "team"]).groupby("season", as_index=False).sum()
    first = df.groupby("season", as_index=False).agg(position=("position", "first"),
                                                     team=("team", lambda t: " / ".join(dict.fromkeys(t))))
    return num.merge(first, on="season").sort_values("season").reset_index(drop=True)
