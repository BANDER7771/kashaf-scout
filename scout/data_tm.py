"""Transfermarkt dataset: download, load (parquet-cached) and player search."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz, process
from unidecode import unidecode

from .config import (CALENDAR_LEAGUES, LEAGUE_STRENGTH, DEFAULT_STRENGTH, POSITION_FALLBACK,
                     SUB_POSITION_GROUP, TM_BASE_URL, TM_DIR, TM_FILES)


def ensure_files(force: bool = False) -> None:
    TM_DIR.mkdir(parents=True, exist_ok=True)
    for name in TM_FILES:
        gz = TM_DIR / f"{name}.csv.gz"
        if gz.exists() and not force:
            continue
        with requests.get(TM_BASE_URL + f"{name}.csv.gz", stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(gz, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        (TM_DIR / f"{name}.parquet").unlink(missing_ok=True)


def _load(name: str, **read_kwargs) -> pd.DataFrame:
    pq = TM_DIR / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    ensure_files()
    df = pd.read_csv(TM_DIR / f"{name}.csv.gz", **read_kwargs)
    df.to_parquet(pq, index=False)
    return df


def normalize_name(s: str) -> str:
    s = unidecode(str(s)).lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def season_of(date: pd.Series, competition: pd.Series) -> pd.Series:
    """Season start year: Jul-Jun seasons, except calendar-year leagues."""
    year = date.dt.year
    calendar = competition.isin(CALENDAR_LEAGUES)
    return np.where(calendar | (date.dt.month >= 7), year, year - 1)


def season_label(season: int, competition: str | None = None) -> str:
    if competition in CALENDAR_LEAGUES:
        return str(season)
    return f"{season}/{str(season + 1)[-2:]}"


@lru_cache(maxsize=1)
def players() -> pd.DataFrame:
    df = _load("players").copy()
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
    df["contract_expiration_date"] = pd.to_datetime(df["contract_expiration_date"], errors="coerce")
    ref = reference_date()
    df["age"] = ((ref - df["date_of_birth"]).dt.days / 365.25).round(1)
    df["contract_years_left"] = ((df["contract_expiration_date"] - ref).dt.days / 365.25).round(2)
    group = df["sub_position"].map(SUB_POSITION_GROUP)
    df["pos_group"] = group.fillna(df["position"].map(POSITION_FALLBACK)).fillna("CM")
    df["league_strength"] = df["current_club_domestic_competition_id"].map(LEAGUE_STRENGTH).fillna(DEFAULT_STRENGTH)
    df["name_norm"] = df["name"].map(normalize_name)
    return df


@lru_cache(maxsize=1)
def competitions() -> pd.DataFrame:
    return _load("competitions")


@lru_cache(maxsize=1)
def domestic_leagues() -> frozenset:
    c = competitions()
    return frozenset(c.loc[c["type"] == "domestic_league", "competition_id"])


@lru_cache(maxsize=1)
def clubs() -> pd.DataFrame:
    return _load("clubs")


@lru_cache(maxsize=1)
def valuations() -> pd.DataFrame:
    df = _load("player_valuations").copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def transfers() -> pd.DataFrame:
    df = _load("transfers").copy()
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    return df


@lru_cache(maxsize=64)
def valuations_for(player_id: int) -> pd.DataFrame:
    """One player's value history; avoids keeping the whole ~130MB table in a small server's RAM."""
    pq = TM_DIR / "player_valuations.parquet"
    if pq.exists():
        df = pd.read_parquet(pq, filters=[("player_id", "==", int(player_id))]).copy()
    else:
        full = valuations()
        df = full[full["player_id"] == player_id].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@lru_cache(maxsize=64)
def transfers_for(player_id: int) -> pd.DataFrame:
    """One player's transfers; avoids keeping the whole ~60MB table in a small server's RAM."""
    pq = TM_DIR / "transfers.parquet"
    if pq.exists():
        df = pd.read_parquet(pq, filters=[("player_id", "==", int(player_id))]).copy()
    else:
        full = transfers()
        df = full[full["player_id"] == player_id].copy()
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def appearances() -> pd.DataFrame:
    df = _load("appearances", usecols=[
        "game_id", "player_id", "player_club_id", "date", "competition_id",
        "yellow_cards", "red_cards", "goals", "assists", "minutes_played"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["season"] = season_of(df["date"], df["competition_id"])
    df["is_league"] = df["competition_id"].isin(domestic_leagues())
    return df


@lru_cache(maxsize=1)
def reference_date() -> pd.Timestamp:
    """'Now' for the dataset = last appearance date (keeps ages/contracts consistent with stats)."""
    txt = TM_DIR / "reference_date.txt"
    if txt.exists():
        return pd.Timestamp(txt.read_text().strip())
    pq = TM_DIR / "appearances.parquet"
    if pq.exists():
        d = pd.read_parquet(pq, columns=["date"])["date"]
    else:
        ensure_files()
        d = pd.read_csv(TM_DIR / "appearances.csv.gz", usecols=["date"])["date"]
    return pd.to_datetime(d).max()


@lru_cache(maxsize=1)
def team_games() -> pd.DataFrame:
    """Number of league games each club played per season (for availability %)."""
    a = appearances()
    a = a[a["is_league"]]
    return (a.groupby(["player_club_id", "season", "competition_id"])["game_id"].nunique()
            .rename("team_games").reset_index())


def search_players(query: str, limit: int = 12, min_season: int = 2022) -> pd.DataFrame:
    df = players()
    pool = df[df["last_season"] >= min_season]
    q = normalize_name(query)
    if not q:
        return pool.head(0)
    matches = process.extract(q, pool["name_norm"], scorer=fuzz.WRatio, limit=limit * 4)
    idx = [m[2] for m in matches if m[1] >= 70]
    res = pool.loc[idx].copy()
    res["score"] = [m[1] for m in matches if m[1] >= 70]
    # Prefer exact-ish matches, then more valuable / recent players.
    res = res.sort_values(["score", "last_season", "market_value_in_eur"], ascending=False)
    return res.head(limit)
