"""Player pool for 16 leagues: style profile per player, look-alike replacements, and search by specification.

Pool = every regular (900+ min, last complete season) from FotMob's league stat lists (Opta), matched to
Transfermarkt for age, market value, contract, foot and height."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import data_fotmob as fm
from . import data_tm as tm
from .config import ARAB_LEAGUES, CACHE_DIR, LEAGUE_STRENGTH, POOL_LEAGUES
from .data_tm import normalize_name

MIN_MINUTES = 900
POOL_FILE = CACHE_DIR / "pool_v2.parquet"

# feature -> (FotMob league stat list, list is a season total that must be turned into per-90)
STYLE_STATS = {
    "xa90": ("expected_assists_per_90", False),
    "chances90": ("total_att_assist", True),
    "bigchances90": ("big_chance_created", True),
    "longballs90": ("accurate_long_balls", False),
    "passes90": ("accurate_pass", False),
    "dribbles90": ("won_contest", False),
    "xg90": ("expected_goals_per_90", False),
    "shots90": ("total_scoring_att", False),
    "tackles90": ("total_tackle", False),
    "interceptions90": ("interception", False),
    "recoveries90": ("ball_recovery", False),
    "press90": ("poss_won_att_3rd", False),
    "clearances90": ("effective_clearance", False),
    "defactions90": ("defensive_contributions", False),
    # Physical tracking - only published for the Premier League in free data.
    "dist90": ("phys_tdc_per_90", False),
    "sprints90": ("phys_sprints_per_90", False),
    "topspeed": ("phys_ts", False),
}
STYLE_FEATURES = [k for k in STYLE_STATS if k not in ("dist90", "sprints90", "topspeed")]
PHYS_FEATURES = ["dist90", "sprints90", "topspeed"]
FEATURE_AR = {
    "xa90": "صناعة الفرص المتوقعة (xA)", "chances90": "الفرص المصنوعة", "bigchances90": "الفرص الكبيرة المصنوعة",
    "longballs90": "الكرات الطويلة الصحيحة", "passes90": "التمريرات الصحيحة", "dribbles90": "المراوغات الناجحة",
    "xg90": "التهديد التهديفي (xG)", "shots90": "التسديدات", "tackles90": "الافتكاكات", "interceptions90": "الاعتراضات",
    "recoveries90": "استعادة الكرة", "press90": "الضغط واستعادة الكرة في الثلث الأخير", "clearances90": "الإبعادات",
    "defactions90": "الأفعال الدفاعية", "dist90": "المسافة المقطوعة", "sprints90": "الانطلاقات السريعة",
    "topspeed": "السرعة القصوى", "rating": "متوسط التقييم",
}
# Same features from a single player's FotMob season stats (fallback when he is not in the pool).
FLAT_MAP = {"xa90": "expected_assists", "chances90": "chances_created", "bigchances90": "big_chance_created_team_title",
            "longballs90": "long_balls_accurate", "passes90": "successful_passes", "dribbles90": "dribbles_succeeded",
            "xg90": "expected_goals", "shots90": "shots", "tackles90": "matchstats.headers.tackles",
            "interceptions90": "interceptions", "recoveries90": "recoveries", "press90": "poss_won_att_3rd_team_title",
            "clearances90": "clearances", "defactions90": "defensive_actions"}


def position_from_code(code: int | None) -> str | None:
    """FotMob line-up grid id -> position group (tens = line, units = lane)."""
    if not code:
        return None
    line, lane = divmod(int(code), 10)
    if line == 1:
        return "GK"
    if line == 3:
        return "FB" if lane in (1, 2, 8, 9) else "CB"
    if line in (4, 5):
        return "FB"
    if line == 6:
        return "DM"
    if line == 7:
        return "W" if lane in (1, 2, 8, 9) else "CM"
    if line == 8:
        return "W" if lane in (1, 2, 8, 9) else "AM"
    return "ST" if line >= 9 else None


def _season_key(season: str) -> str:
    return season.replace("/", "-")


def _stat_list(league_id: int, season: str, stat: str, urls: dict) -> dict[int, tuple[float, int]]:
    url = urls.get(stat)
    if not url:
        return {}
    data = fm._get(url, f"stat_{league_id}_{_season_key(season)}_{stat}", 24 * 7)
    try:
        rows = data["TopLists"][0]["StatList"]
    except (TypeError, KeyError, IndexError):
        return {}
    return {e["ParticiantId"]: (float(e["StatValue"]), e.get("MinutesPlayed") or 0) for e in rows}


def _league(league_id: int, code: str, calendar: bool, season_start: int) -> pd.DataFrame:
    season = str(season_start) if calendar else f"{season_start}/{season_start + 1}"
    league = fm._get(f"{fm.BASE}/leagues?id={league_id}&season={season}", f"league_{league_id}_{_season_key(season)}", 24 * 7)
    if not league:
        return pd.DataFrame()
    urls = {x.get("name"): x.get("fetchAllUrl") for x in league.get("stats", {}).get("players", [])}
    rating = fm._get(urls.get("rating"), f"rating_{league_id}_{_season_key(season)}", 24 * 7) if urls.get("rating") else None
    if not rating:
        return pd.DataFrame()
    rows = []
    for e in rating["TopLists"][0]["StatList"]:
        if (e.get("MinutesPlayed") or 0) < MIN_MINUTES:
            continue
        rows.append({"fotmob_id": e["ParticiantId"], "name": e["ParticipantName"], "team": e["TeamName"],
                     "league": code, "rating": float(e["StatValue"]), "minutes": e["MinutesPlayed"],
                     "matches": e.get("MatchesPlayed"), "pos_code": (e.get("Positions") or [None])[0]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    with ThreadPoolExecutor(8) as ex:
        lists = dict(zip(STYLE_STATS, ex.map(lambda s: _stat_list(league_id, season, STYLE_STATS[s][0], urls), STYLE_STATS)))
    for feat, (stat, is_total) in STYLE_STATS.items():
        vals = lists[feat]
        if not vals:
            df[feat] = np.nan
            continue
        if is_total:
            df[feat] = [vals[i][0] / max(vals[i][1], 1) * 90 if i in vals else 0.0 for i in df["fotmob_id"]]
        else:
            # Not listed = below the list's floor; for event counts that is effectively zero.
            fill = np.nan if feat in PHYS_FEATURES else 0.0
            df[feat] = [vals[i][0] if i in vals else fill for i in df["fotmob_id"]]
    return df


def _last_complete_season() -> int:
    now = pd.Timestamp.now()
    return now.year - 1 if now.month >= 7 else now.year - 2


@lru_cache(maxsize=1)
def pool() -> pd.DataFrame:
    season = _last_complete_season()
    if POOL_FILE.exists() and (pd.Timestamp.now() - pd.Timestamp(POOL_FILE.stat().st_mtime, unit="s")).days < 7:
        return pd.read_parquet(POOL_FILE)
    jobs = [(lid, code, cal, season + 1 if cal else season) for lid, (code, cal) in POOL_LEAGUES.items()]
    with ThreadPoolExecutor(4) as ex:
        frames = [f for f in ex.map(lambda j: _league(*j), jobs) if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["pos_group"] = df["pos_code"].map(position_from_code)
    df["strength"] = df["league"].map(LEAGUE_STRENGTH).fillna(0.6)
    # A 7.4 in a weaker league is worth less than a 7.4 in the Premier League.
    df["rating_adj"] = df["rating"] - (1 - df["strength"]) * 1.2
    df = _attach_transfermarkt(df)
    POOL_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(POOL_FILE, index=False)
    return df


def _attach_transfermarkt(df: pd.DataFrame) -> pd.DataFrame:
    p = tm.players()
    p = p[p["last_season"] >= p["last_season"].max() - 1]
    by_name = {k: g for k, g in p.groupby("name_norm")}
    cols = ["player_id", "age", "market_value_in_eur", "contract_expiration_date", "current_club_name",
            "pos_group", "sub_position", "image_url", "country_of_citizenship", "foot", "height_in_cm"]
    matched = []
    for r in df.itertuples():
        cands = by_name.get(normalize_name(r.name))
        if cands is None:
            matched.append({c: None for c in cols})
            continue
        best = max(cands.itertuples(), key=lambda c: fuzz.partial_ratio(normalize_name(r.team), normalize_name(c.current_club_name or "")))
        matched.append({c: getattr(best, c) for c in cols})
    m = pd.DataFrame(matched).rename(columns={"pos_group": "tm_pos_group"})
    out = pd.concat([df.reset_index(drop=True), m], axis=1)
    # Transfermarkt's position label is more reliable than FotMob's line-up grid id (wingers often sit on the
    # forward line in a 4-3-3 and would read as strikers).
    out["pos_group"] = out["tm_pos_group"].fillna(out["pos_group"])
    return out


# ---------------------------------------------------------------- style space

@lru_cache(maxsize=1)
def zscores() -> pd.DataFrame:
    """Every style feature standardised within the player's position group (a CB is compared with CBs)."""
    df = pool()
    feats = STYLE_FEATURES + PHYS_FEATURES + ["rating_adj"]
    z = df[["pos_group"] + feats].copy()
    for f in feats:
        g = df.groupby("pos_group")[f]
        z[f] = (df[f] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return z


RELATED_POSITIONS = {"CM": ["CM", "DM", "AM"], "DM": ["DM", "CM"], "AM": ["AM", "CM", "W"], "W": ["W", "AM"],
                     "ST": ["ST"], "FB": ["FB"], "CB": ["CB"], "GK": ["GK"]}


def _target_raw(pos_group: str, fotmob_id: int | None, flat: dict | None) -> pd.Series:
    df = pool()
    row = df[df["fotmob_id"] == fotmob_id] if fotmob_id is not None else df.head(0)
    if not row.empty:
        return row.iloc[0][STYLE_FEATURES + PHYS_FEATURES].astype(float)
    out = {f: float((flat or {}).get(key, {}).get("per90")) for f, key in FLAT_MAP.items()
           if (flat or {}).get(key, {}).get("per90") is not None}
    return pd.Series(out, dtype=float)


def similar_style(pos_group: str, fotmob_id: int | None, flat: dict | None) -> pd.Series:
    """Cosine similarity (0-100) of each candidate's style to the target's. Everyone is standardised with the
    target position's averages, so a CM and a DM are compared on the same scale."""
    df = pool()
    target = _target_raw(pos_group, fotmob_id, flat)
    base = df[df["pos_group"] == pos_group]
    cand = df[df["pos_group"].isin(RELATED_POSITIONS.get(pos_group, [pos_group]))]
    feats = [f for f in STYLE_FEATURES + PHYS_FEATURES if f in target.index and pd.notna(target[f]) and base[f].std() > 0]
    if len(feats) < 5 or cand.empty:
        return pd.Series(dtype=float)
    mu, sd = base[feats].mean(), base[feats].std()
    a = ((target[feats] - mu) / sd).to_numpy(dtype=float)
    m = ((cand[feats] - mu) / sd).to_numpy(dtype=float)
    mask = ~np.isnan(m)
    m = np.where(mask, m, 0.0)
    num = (m * a).sum(axis=1)
    den = np.sqrt((m ** 2).sum(axis=1)) * np.sqrt(((a * mask) ** 2).sum(axis=1))
    cos = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    sim = pd.Series(np.clip(cos, 0, 1) * 100, index=cand.index)
    # A different (related) position is a slightly worse like-for-like replacement.
    return sim - np.where(cand["pos_group"] == pos_group, 0, 6)


def alternatives(pos_group: str, rating_adj: float | None, base_value: float, age: float, fotmob_id: int | None,
                 flat: dict | None = None, scope: str = "all", league: str | None = None, n: int = 8) -> pd.DataFrame:
    """Players with the most similar playing style at the same position, preferring better value for money."""
    df = pool()
    if df.empty:
        return df
    sim = similar_style(pos_group, fotmob_id, flat)
    if sim.empty:
        return df.head(0)
    c = df.loc[sim.index].assign(similarity=sim)
    c = c[c["market_value_in_eur"].notna() & (c["market_value_in_eur"] > 0) & (c["fotmob_id"] != fotmob_id)]
    if scope == "league" and league:
        c = c[c["league"] == league]
    elif scope == "arab":
        c = c[c["league"].isin(ARAB_LEAGUES)]
    budget = max(base_value * (1.5 if age >= 31 else 1.2), 3_000_000)
    c = c[(c["market_value_in_eur"] <= budget) & (c["age"].fillna(99) <= max(age + 2, 31)) & (c["similarity"] >= 60)]
    if c.empty:
        return c.assign(score=[], why=[])
    ref = rating_adj if rating_adj is not None else c["rating_adj"].median()
    c["value_ratio"] = c["market_value_in_eur"] / max(base_value, 1)
    c["rating_gain"] = c["rating_adj"] - ref
    # Style first (a 10-point similarity gap outweighs most rating gaps), then level, then price and age.
    c["score"] = (c["similarity"] * 0.05 + c["rating_gain"].clip(-0.6, 0.6) * 0.8
                  - np.log(c["value_ratio"].clip(lower=0.05)) * 0.15 + np.clip(age - c["age"], -4, 8) * 0.03)
    c["why"] = [_why(r, age) for r in c.itertuples()]
    return c.sort_values("score", ascending=False).head(n)


def _why(r, age: float) -> str:
    parts = [f"تشابه الأسلوب {r.similarity:.0f}%"]
    if r.rating_gain >= 0.05:
        parts.append(f"أداء أعلى (+{r.rating_gain:.2f})")
    elif r.rating_gain >= -0.05:
        parts.append("أداء مماثل")
    if r.value_ratio < 0.95:
        parts.append(f"أرخص بـ {1 - r.value_ratio:.0%}")
    elif r.value_ratio > 1.05:
        parts.append(f"أغلى بـ {r.value_ratio - 1:.0%}")
    if pd.notna(r.age) and age - r.age >= 2:
        parts.append(f"أصغر بـ {age - r.age:.0f} سنوات")
    return " · ".join(parts)


# ---------------------------------------------------------------- search by specification

def search(spec: dict, n: int = 20) -> pd.DataFrame:
    """Rank pool players against a specification: weighted style + level fit (0-100, percentile among all
    players in the requested positions), then hard filters (age, price, leagues, foot, contract...)."""
    df = pool()
    if df.empty:
        return df
    z = zscores()
    c = df[df["pos_group"].isin(spec["positions"])] if spec.get("positions") else df
    weights = {k: float(v) for k, v in (spec.get("style_weights") or {}).items()
               if k in STYLE_FEATURES + PHYS_FEATURES and v}
    zc = z.loc[c.index]
    style = pd.Series(0.0, index=c.index)
    for f, w in weights.items():
        style = style + zc[f].fillna(-0.5 if f in PHYS_FEATURES else 0.0) * w
    if weights:
        style = style / sum(abs(w) for w in weights.values())
    q = float(spec.get("quality_weight", 0.35))
    raw = (1 - q) * style + q * zc["rating_adj"].fillna(0)
    c = c.assign(match=(raw.rank(pct=True) * 100).round(), style_z=style)

    if spec.get("age_min"):
        c = c[c["age"] >= spec["age_min"]]
    if spec.get("age_max"):
        c = c[c["age"] <= spec["age_max"]]
    if spec.get("max_value_eur"):
        c = c[c["market_value_in_eur"].fillna(0) <= spec["max_value_eur"]]
    if spec.get("min_value_eur"):
        c = c[c["market_value_in_eur"].fillna(0) >= spec["min_value_eur"]]
    scope = spec.get("scope") or "all"
    if scope == "arab":
        c = c[c["league"].isin(ARAB_LEAGUES)]
    elif scope == "europe":
        c = c[~c["league"].isin(ARAB_LEAGUES | {"MLS1", "BRA1"})]
    if spec.get("leagues"):
        c = c[c["league"].isin(spec["leagues"])]
    if spec.get("foot") in ("left", "right"):
        c = c[c["foot"] == spec["foot"]]
    if spec.get("min_height"):
        c = c[c["height_in_cm"].fillna(0) >= spec["min_height"]]
    if spec.get("nationality"):
        c = c[c["country_of_citizenship"].fillna("").str.contains(spec["nationality"], case=False)]
    if spec.get("contract_expiring"):
        exp = pd.to_datetime(c["contract_expiration_date"], errors="coerce")
        c = c[exp <= pd.Timestamp.now() + pd.Timedelta(days=270)]
    return c.sort_values(["match", "rating_adj"], ascending=False).head(n)
