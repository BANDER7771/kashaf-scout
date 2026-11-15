"""FotMob (Opta-based deep stats for 100+ leagues incl. Saudi Pro League) with an on-disk cache.

Public JSON endpoints used by fotmob.com - for private testing only."""
from __future__ import annotations

import json
import re
import time

import pandas as pd
import requests
from rapidfuzz import fuzz

from .config import CACHE_DIR
from .data_tm import normalize_name

BASE = "https://www.fotmob.com/api/data"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/128.0 Safari/537.36", "Accept": "application/json"}

POSITION_KEY_GROUP = {
    "keeper": "GK", "goalkeeper": "GK",
    "centerback": "CB", "leftcenterback": "CB", "rightcenterback": "CB",
    "leftback": "FB", "rightback": "FB", "leftwingback": "FB", "rightwingback": "FB",
    "centerdefensivemidfielder": "DM", "defensivemidfielder": "DM",
    "centermidfielder": "CM", "leftcentermidfielder": "CM", "rightcentermidfielder": "CM",
    "centerattackingmidfielder": "AM", "attackingmidfielder": "AM",
    "leftmidfielder": "W", "rightmidfielder": "W", "leftwinger": "W", "rightwinger": "W",
    "striker": "ST", "centerforward": "ST", "secondstriker": "ST",
}
NON_LEAGUE = re.compile("|".join([
    "(?<![a-z])cup(?![a-z])", "champions league", "world cup", "friendl", "nations league", "qualif", "super ?cup",
    "supercopa", "supercoppa", "europa", "conference league", "copa", "(?<![a-z])euro(?![a-z])", "(?<![a-z])afc(?![a-z])",
    "(?<![a-z])caf(?![a-z])", "concacaf", "olympic", "play-?offs?", "shield", "trophy", "pokal", "coupe", "coppa",
    "asian games"]), re.I)


def _get(url: str, cache_key: str, ttl_hours: float):
    cache = CACHE_DIR / "fotmob" / f"{cache_key}.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < ttl_hours * 3600:
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
    if data is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def search(name: str, club_hint: str | None = None) -> int | None:
    term = requests.utils.quote(name)
    res = _get(f"{BASE}/search/suggest?term={term}&lang=en", f"search_{normalize_name(name).replace(' ', '_')}", 24 * 7)
    if not res:
        return None
    cands = [s for grp in res for s in grp.get("suggestions", []) if s.get("type") == "player"]
    if not cands:
        return None
    target = normalize_name(name)

    def score(s):
        sc = fuzz.WRatio(target, normalize_name(s["name"]))
        if club_hint and s.get("teamName"):
            sc += 10 if fuzz.partial_ratio(normalize_name(club_hint), normalize_name(s["teamName"])) > 70 else 0
        return sc
    best = max(cands, key=score)
    return int(best["id"]) if score(best) >= 80 else None


def player(fotmob_id: int) -> dict | None:
    return _get(f"{BASE}/playerData?id={fotmob_id}", f"player_{fotmob_id}", 12)


def season_stats(fotmob_id: int, entry_id: str) -> dict | None:
    return _get(f"{BASE}/playerStats?playerId={fotmob_id}&seasonId={entry_id}", f"stats_{fotmob_id}_{entry_id}", 24)


def league_entries(p: dict) -> list[dict]:
    """Domestic-league season entries with deep stats, newest first."""
    out = []
    for season in p.get("statSeasons", []):
        for t in season.get("tournaments", []):
            if t.get("hasDeepStats") and not NON_LEAGUE.search(t.get("name", "")):
                out.append({"season": season["seasonName"], "league": t["name"], "entry_id": t["entryId"]})
                break
    return out


def flatten_stats(stats: dict) -> dict:
    """{stat_id: {value, per90, pct}} from a playerStats payload (+ minutes / rating from the top card)."""
    flat = {}
    for group in (stats or {}).get("statsSection", {}).get("items", []):
        for item in group.get("items", []):
            flat[item["localizedTitleId"]] = {"title": item.get("title"), "value": item.get("statValue"),
                                              "per90": item.get("per90"), "pct": item.get("percentileRankPer90"),
                                              "format": item.get("statFormat")}
    for item in (stats or {}).get("topStatCard", {}).get("items", []):
        flat.setdefault(item["localizedTitleId"], {"title": item.get("title"), "value": item.get("statValue"),
                                                   "per90": item.get("per90"), "pct": item.get("percentileRankPer90")})
    return flat


def minutes(flat: dict) -> int:
    try:
        return int(float(flat.get("minutes_played", {}).get("value") or 0))
    except ValueError:
        return 0


def best_season_stats(fotmob_id: int, p: dict, min_minutes: int = 900) -> tuple[dict, dict] | tuple[None, None]:
    """Latest domestic-league season with enough minutes (falls back to the busiest of the last two)."""
    tried = []
    for e in league_entries(p)[:3]:
        flat = flatten_stats(season_stats(fotmob_id, e["entry_id"]))
        if not flat:
            continue
        tried.append((e, flat))
        if minutes(flat) >= min_minutes:
            return e, flat
    if tried:
        e, flat = max(tried[:2], key=lambda t: minutes(t[1]))
        return e, flat
    return None, None


def positions(p: dict) -> tuple[str | None, list[str]]:
    """Main position + positions he has genuinely played (at least 20% of the main one's games, min 3)."""
    desc = (p or {}).get("positionDescription") or {}
    items = sorted(desc.get("positions", []), key=lambda x: -x.get("occurences", 0))
    main_item = next((x for x in items if x.get("isMainPosition")), items[0] if items else None)
    if not main_item:
        return None, []
    main = POSITION_KEY_GROUP.get(main_item.get("strPos", {}).get("key", ""))
    floor = max(3, 0.2 * (main_item.get("occurences") or 0))
    alts = []
    for pos in items:
        g = POSITION_KEY_GROUP.get(pos.get("strPos", {}).get("key", ""))
        if g and g != main and g not in alts and (pos.get("occurences") or 0) >= floor:
            alts.append(g)
    return main, alts


def team_color(p: dict) -> str | None:
    return (((p or {}).get("primaryTeam") or {}).get("teamColors") or {}).get("color")


def _int(x) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def career_seasons(p: dict) -> pd.DataFrame:
    """Season-by-season domestic league record (all leagues FotMob covers), with average rating."""
    rows = []
    for s in p.get("careerHistory", {}).get("careerItems", {}).get("senior", {}).get("seasonEntries", []):
        for t in s.get("tournamentStats", []):
            if t.get("isFriendly") or NON_LEAGUE.search(t.get("leagueName", "")):
                continue
            rating = (t.get("rating") or {}).get("rating")
            try:
                rating = float(rating) if rating not in (None, "", "0") else None
            except ValueError:
                rating = None
            rows.append({"label": s["seasonName"], "club": s.get("team"), "league": t["leagueName"],
                         "games": _int(t.get("appearances")), "goals": _int(t.get("goals")),
                         "assists": _int(t.get("assists")),
                         "rating": rating})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["season"] = df["label"].str[:4].astype(int)
    return df.sort_values(["season", "games"]).drop_duplicates(["label", "club", "league"]).reset_index(drop=True)


def match_ratings(p: dict) -> pd.DataFrame:
    rows = []
    for m in p.get("recentMatches", []):
        r = (m.get("ratingProps") or {}).get("rating")
        if m.get("onBench") or not r or (m.get("minutesPlayed") or 0) < 30:
            continue
        rows.append({"date": pd.to_datetime(m["matchDate"]["utcTime"]).tz_localize(None), "league": m.get("leagueName"),
                     "opponent": m.get("opponentTeamName"), "minutes": m.get("minutesPlayed"),
                     "goals": m.get("goals", 0), "assists": m.get("assists", 0), "rating": float(r)})
    return pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame()


def market_values(p: dict) -> pd.DataFrame:
    vals = (p.get("marketValues") or {}).get("values", [])
    if not vals:
        return pd.DataFrame()
    df = pd.DataFrame(vals)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[["date", "value", "teamName"]].rename(columns={"value": "market_value_in_eur", "teamName": "current_club_name"})


def info(p: dict) -> dict:
    out = {}
    for item in p.get("playerInformation", []):
        out[item.get("title")] = (item.get("value") or {}).get("fallback")
    return out


def season_flats(fotmob_id: int, p: dict, n: int = 2, min_minutes: int = 450) -> list[tuple[dict, dict]]:
    """Up to n most recent domestic-league seasons with deep stats and real minutes (newest first)."""
    out = []
    for e in league_entries(p)[:4]:
        flat = flatten_stats(season_stats(fotmob_id, e["entry_id"]))
        if flat and minutes(flat) >= min_minutes:
            out.append((e, flat))
        if len(out) >= n:
            break
    return out


def league_id_for(p: dict, season: str, league_name: str) -> int | None:
    for s in p.get("statSeasons", []):
        if s.get("seasonName") == season:
            for t in s.get("tournaments", []):
                if t.get("name") == league_name:
                    return t.get("tournamentId")
    return None


def team_context(league_id: int | None, season: str, team_id: int | None, team_name: str | None) -> dict | None:
    """Where the player's team finished / sits in the league table that season."""
    if not league_id:
        return None
    lg = _get(f"{BASE}/leagues?id={league_id}&season={season}", f"league_{league_id}_{season.replace('/', '-')}", 24)
    try:
        rows = lg["table"][0]["data"]["table"]["all"]
    except (TypeError, KeyError, IndexError):
        return None
    for r in rows:
        if r.get("id") == team_id or (team_name and fuzz.partial_ratio(normalize_name(team_name), normalize_name(r["name"])) > 85):
            return {"team": r["name"], "position": r.get("idx"), "teams": len(rows), "played": r.get("played"),
                    "wins": r.get("wins"), "draws": r.get("draws"), "losses": r.get("losses"),
                    "goals": r.get("scoresStr"), "points": r.get("pts")}
    return None


def positions_detail(p: dict) -> list[dict]:
    desc = (p or {}).get("positionDescription") or {}
    return [{"position": x.get("strPos", {}).get("label"), "games": x.get("occurences"), "main": x.get("isMainPosition"),
             "group": POSITION_KEY_GROUP.get(x.get("strPos", {}).get("key", ""))}
            for x in sorted(desc.get("positions", []), key=lambda x: -x.get("occurences", 0))]


INJURY_AR = {"tendon": "إصابة في الوتر", "hamstring": "إصابة في العضلة الخلفية", "knee": "إصابة في الركبة",
             "ankle": "إصابة في الكاحل", "muscle": "إصابة عضلية", "groin": "إصابة في أعلى الفخذ",
             "calf": "إصابة في عضلة الساق", "thigh": "إصابة في الفخذ", "back": "إصابة في الظهر", "foot": "إصابة في القدم",
             "shoulder": "إصابة في الكتف", "achilles": "إصابة في وتر أخيل", "cruciate": "قطع في الرباط الصليبي",
             "ligament": "إصابة في الأربطة", "knock": "كدمة", "illness": "مرض", "head": "إصابة في الرأس",
             "hip": "إصابة في الورك", "fracture": "كسر"}
RETURN_AR = {"back in training": "عاد للتدريبات", "doubtful": "مشكوك في مشاركته", "unknown": "غير معروف"}


def injury(p: dict) -> dict | None:
    inj = (p or {}).get("injuryInformation")
    if not inj:
        return None
    name = inj.get("name") or ""
    ar = next((v for k, v in INJURY_AR.items() if k in name.lower()), name)
    ret = (inj.get("expectedReturn") or {}).get("expectedReturnFallback") or ""
    return {"injury": ar, "expected_return": RETURN_AR.get(str(ret).lower(), ret),
            "updated": ((inj.get("lastUpdated") or {}).get("utcTime") or "")[:10]}


def trophies(p: dict) -> list[str]:
    out = []
    for team in ((p or {}).get("trophies") or {}).get("playerTrophies", []):
        for t in team.get("tournaments", []):
            if t.get("seasonsWon"):
                out.append(f"{t['leagueName']} ×{len(t['seasonsWon'])} ({team.get('teamName')})")
    return out


def team_id(p: dict) -> int | None:
    return ((p or {}).get("primaryTeam") or {}).get("teamId")


def photo_url(fotmob_id: int) -> str:
    return f"https://images.fotmob.com/image_resources/playerimages/{fotmob_id}.png"


def team_logo_url(team_id: int) -> str:
    return f"https://images.fotmob.com/image_resources/logo/teamlogo/{team_id}.png"
