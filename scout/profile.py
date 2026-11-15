"""Builds a full player profile: data layer first, then the DeepSeek + web-research layer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ai, career, fit, metrics, valuation, web  # noqa: F401 (fit kept for SYSTEMS catalogue)
from . import data_fotmob as fm
from . import data_tm as tm
from . import data_understat as us
from .config import LEAGUE_NAMES_AR, LEAGUE_STRENGTH, DEFAULT_STRENGTH, POSITION_NAMES_AR, UNDERSTAT_LEAGUES

def _understat(player: pd.Series, seasons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    us_rows = seasons[seasons["competition_id"].isin(UNDERSTAT_LEAGUES)]
    if us_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), None
    last = us_rows.iloc[-1]
    uid = us.find_player(player["name"], UNDERSTAT_LEAGUES[last["competition_id"]], int(last["season"]), last["club"])
    data = us.player_data(uid) if uid else None
    if not data:
        return pd.DataFrame(), pd.DataFrame(), None
    return us.player_matches(data), us.player_seasons(data), uid


def _fotmob(player: pd.Series) -> dict:
    out = {"id": None, "raw": None, "entry": None, "flat": {}, "main_pos": None, "alt_pos": [],
           "seasons": pd.DataFrame(), "matches": pd.DataFrame(), "values": pd.DataFrame(), "color": None,
           "flats": [], "team_ctx": None, "positions_detail": [], "injury": None, "trophies": [], "team_id": None}
    fid = fm.search(player["name"], player["current_club_name"])
    raw = fm.player(fid) if fid else None
    if not raw:
        return out
    entry, flat = fm.best_season_stats(fid, raw)
    main, alts = fm.positions(raw)
    flats = fm.season_flats(fid, raw, n=2)
    team_ctx = None
    if entry:
        team_ctx = fm.team_context(fm.league_id_for(raw, entry["season"], entry["league"]), entry["season"],
                                   fm.team_id(raw), player.get("current_club_name"))
    out.update(id=fid, raw=raw, entry=entry, flat=flat or {}, main_pos=main, alt_pos=alts,
               seasons=fm.career_seasons(raw), matches=fm.match_ratings(raw), values=fm.market_values(raw),
               color=fm.team_color(raw), flats=flats, team_ctx=team_ctx, positions_detail=fm.positions_detail(raw),
               injury=fm.injury(raw), trophies=fm.trophies(raw), team_id=fm.team_id(raw))
    return out


def build_base(player_id: int) -> dict:
    p = tm.players().set_index("player_id").loc[player_id]
    f = _fotmob(p)
    # FotMob's main position comes from actual line-ups, so prefer it over Transfermarkt's label.
    pos = f["main_pos"] or p["pos_group"]
    tm_seasons = metrics.player_league_seasons(player_id)
    us_matches, us_seasons, us_id = _understat(p, tm_seasons)
    season_row = metrics.pick_rating_season(tm_seasons)

    season_rating, league_code = None, p["current_club_domestic_competition_id"]
    if f["flat"] and fm.minutes(f["flat"]) >= 450:
        strength, league_code = metrics.fotmob_league_strength(f["entry"]["league"], league_code)
        data_ratings = metrics.fotmob_ratings(pos, f["flat"], strength, f["entry"]["season"], f["entry"]["league"])
        peer = metrics.fotmob_peer_metrics(pos, f["flat"])
        rating_season = f"{f['entry']['league']} {f['entry']['season']} ({fm.minutes(f['flat'])} دقيقة)"
        try:
            season_rating = float(f["flat"].get("rating", {}).get("value"))
        except (TypeError, ValueError):
            season_rating = None
        # Understat adds a multi-season finishing-vs-xG signal for top-5 league attackers.
        us_r = metrics.data_ratings(pos, season_row, us_seasons) if not us_seasons.empty else {}
        if "finishing" in us_r and "finishing" in data_ratings:
            data_ratings["finishing"] = {
                "value": round(0.6 * data_ratings["finishing"]["value"] + 0.4 * us_r["finishing"]["value"]),
                "detail": data_ratings["finishing"]["detail"] + " + Understat: " + us_r["finishing"]["detail"]}
        if "composure" in us_r:
            data_ratings.setdefault("composure", us_r["composure"])
    else:
        data_ratings = metrics.data_ratings(pos, season_row, us_seasons)
        peer = metrics.peer_metrics(pos, season_row, us_seasons)
        rating_season = None if season_row is None else f"{season_row['label']} ({int(season_row['minutes'])} دقيقة)"

    seasons = career.unified_seasons(tm_seasons, f["seasons"])
    mv = career.market_values(player_id, f["values"])
    tr = career.player_transfers(player_id)
    rows = career.timeline(pos, seasons, mv, tr)
    sci = float(f["values"]["market_value_in_eur"].iloc[-1]) if not f["values"].empty else None
    val = valuation.fair_value(player_id, scisports_value=sci)
    strength = LEAGUE_STRENGTH.get(league_code, DEFAULT_STRENGTH)
    cons = career.consistency(player_id, pos, seasons, us_matches, f["matches"], mv)
    base = {
        "player_id": player_id,
        "player": p.to_dict(),
        "pos_group": pos,
        "league_code": league_code,
        "team_color": f["color"],
        # FotMob head-shots are transparent cut-outs (sharp in the round card); Transfermarkt's large portrait is the fallback.
        "photo": fm.photo_url(f["id"]) if f["id"] else str(p.get("image_url") or "").replace("/header/", "/big/"),
        "team_logo": fm.team_logo_url(f["team_id"]) if f["team_id"] else None,
        "fotmob_id": f["id"],
        "fotmob_positions": {"main": pos, "alternatives": [a for a in f["alt_pos"] if a != pos]},
        "fotmob_matches": f["matches"],
        "us_matches": us_matches,
        "understat_id": us_id,
        "rating_season": rating_season,
        "season_rating": season_rating,
        "rating_adj": None if season_rating is None else season_rating - (1 - strength) * 1.2,
        "data_ratings": data_ratings,
        "peer_metrics": peer,
        "seasons": rows,
        "market_values": mv,
        "transfers": tr,
        "consistency": cons,
        "traits_data": metrics.trait_signals(pos, f["flat"], cons["score"]),
        "league_strength": strength,
        "season_stats": [{"season": e["season"], "league": e["league"], "stats": fl} for e, fl in f["flats"]],
        "team_context": f["team_ctx"],
        "positions_detail": f["positions_detail"],
        "injury": f["injury"],
        "trophies": f["trophies"],
        "us_seasons": us_seasons,
        "trend": career.trend(seasons, mv),
        "valuation": val,
    }
    base["flags"] = [f"{s['label']} ({s['club']}): {', '.join(s['tags'])}" for s in rows if s["tags"]]
    return base


PILLARS = {
    "الحسم": ["end_product"],
    "الصناعة": ["chance_creation", "progression"],
    "التقدم بالكرة": ["ball_carrying", "press_resistance"],
    "الدفاع": ["pressing", "defensive_discipline"],
    "البدني": ["physical_effort", "duels"],
    "الثبات": ["consistency"],
}


def _merge_traits(base: dict, ai_out: dict | None) -> list[dict]:
    """Measured baseline and the AI's evidence-based judgement, weighted per trait (see TRAIT_DATA_WEIGHT)."""
    data = base["traits_data"]
    ai_traits = {t["key"]: t for t in (ai_out or {}).get("traits", [])}
    pros_keys = {x["trait"] for x in (ai_out or {}).get("pros", []) if x.get("trait")}
    cons_keys = {x["trait"] for x in (ai_out or {}).get("cons", []) if x.get("trait")}
    out = []
    for key, label in metrics.TRAITS.items():
        d, a = data.get(key), ai_traits.get(key)
        if d and a and a.get("score") is not None:
            w = metrics.TRAIT_DATA_WEIGHT.get(key, 0.5)
            score = round(w * d["data_pct"] + (1 - w) * a["score"])
        elif d:
            score = d["data_pct"]
        elif a and a.get("score") is not None:
            score = a["score"]
        else:
            continue
        # The written pros/cons carry the evidence (reports on running, intensity...), so the bar must agree with them.
        if key in cons_keys and key not in pros_keys:
            score = min(score, 40)
        elif key in pros_keys and key not in cons_keys:
            score = max(score, 65)
        verdict = "قوة" if score >= 65 else "ضعف" if score <= 40 else "متوسط"
        out.append({"key": key, "label": label, "score": score, "verdict": verdict,
                    "data_pct": d["data_pct"] if d else None, "basis": d["basis"] if d else "",
                    "explanation": (a or {}).get("explanation", "")})
    return sorted(out, key=lambda t: t["score"], reverse=True)


def _rating_level(rating_adj: float | None) -> int | None:
    """League-adjusted average match rating on the 0-99 scale (6.0 -> 50, 7.0 -> 69, 8.0 -> 88)."""
    if rating_adj is None:
        return None
    return int(np.clip(round(50 + (rating_adj - 6.0) * 19), 40, 95))


def combine(base: dict, ai_out: dict | None) -> dict:
    """Merge measured data with the AI analysis -> overall, card pillars, traits, positions played."""
    traits = _merge_traits(base, ai_out)
    by_key = {t["key"]: t["score"] for t in traits}
    pillars = {}
    for name, keys in PILLARS.items():
        vals = [by_key[k] for k in keys if k in by_key]
        pillars[name] = round(sum(vals) / len(vals)) if vals else None
    data_level = _rating_level(base.get("rating_adj"))
    ai_overall = (ai_out or {}).get("overall")
    if data_level is not None and ai_overall:
        overall = round(0.5 * data_level + 0.5 * ai_overall)
    else:
        overall = data_level or ai_overall
    played = {}
    for p in base.get("positions_detail") or []:
        g = p.get("group")
        if g:
            played[g] = played.get(g, 0) + (p.get("games") or 0)
    pos = base["pos_group"]
    # Only positions he has genuinely played (a couple of cameo games elsewhere is not a position).
    floor = max(3, 0.15 * max(played.values(), default=0))
    positions = [{"group": g, "games": n, "natural": g == pos} for g, n in played.items() if g == pos or n >= floor]
    if pos not in played:
        positions.insert(0, {"group": pos, "games": None, "natural": True})
    positions.sort(key=lambda x: (not x["natural"], -(x["games"] or 0)))
    return {
        "traits": traits,
        "pillars": pillars,
        "overall": overall,
        "potential": (ai_out or {}).get("potential"),
        "positions": positions,
        "alternatives": [a for a in base["fotmob_positions"]["alternatives"] if a != pos],
    }


def _stats_for_ai(flat: dict) -> dict:
    """Every deep stat as one unambiguous Arabic line, so values and percentile ranks can't be confused."""
    return {metrics.stat_label(k, s.get("title")): metrics.describe_stat(k, s) for k, s in flat.items()}


AGE_PEAK = {"ST": (25, 30), "W": (24, 29), "AM": (25, 30), "CM": (25, 31), "DM": (26, 31), "FB": (25, 30),
            "CB": (26, 32), "GK": (28, 34)}


def signature(base: dict) -> dict:
    """What most defines the player in the data: his highest-ranked traits and stats (latest full season)."""
    traits = sorted(((t["label"], t["data_pct"]) for k, t in base["traits_data"].items() if k != "consistency"),
                    key=lambda x: -x[1])[:3]
    full = [s for s in base["season_stats"] if fm.minutes(s["stats"]) >= 900] or base["season_stats"]
    stats = []
    if full:
        flat = full[0]["stats"]
        ranked = sorted(((k, v) for k, v in flat.items() if v.get("pct") is not None
                         and k not in ("rating", "matches_uppercase", "player_started_matches", "minutes_played",
                                       "yellow_cards", "red_cards", "fouls", "clean_sheet_team_title",
                                       "goals_conceded_while_on_pitch", "expected_goals_against_while_on_pitch")),
                        key=lambda kv: -float(kv[1]["pct"]))[:5]
        stats = [f"{metrics.stat_label(k, v.get('title'))}: أفضل من {float(v['pct']):.0f}% من لاعبي مركزه" for k, v in ranked]
    return {"top_traits": [f"{lbl} ({pct})" for lbl, pct in traits], "top_stats": stats}


def analysis_data(base: dict) -> dict:
    p = base["player"]
    v = base["valuation"]
    age = float(p["age"]) if pd.notna(p["age"]) else None
    lo, hi = AGE_PEAK.get(base["pos_group"], (25, 30))
    mv = base["market_values"]
    mv_hist = []
    if not mv.empty:
        yearly = mv.assign(year=mv["date"].dt.year).drop_duplicates("year", keep="last")
        mv_hist = [{"year": int(r.year), "value_eur": float(r.market_value_in_eur), "club": r.current_club_name}
                   for r in yearly.itertuples()]
    matches = base["fotmob_matches"]
    recent = [] if matches.empty else [
        {"date": str(r.date.date()), "competition": r.league, "opponent": r.opponent, "minutes": r.minutes,
         "goals": r.goals, "assists": r.assists, "rating": r.rating} for r in matches.tail(15).itertuples()]
    us_seasons = base.get("us_seasons")
    xg = [] if us_seasons is None or us_seasons.empty else [
        {"season": f"{int(r.season)}/{str(int(r.season) + 1)[-2:]}", "team": r.team, "games": int(r.games),
         "minutes": int(r.time), "goals": int(r.goals), "npg": int(r.npg), "xG": round(r.xG, 2),
         "npxG": round(r.npxG, 2), "assists": int(r.assists), "xA": round(r.xA, 2), "shots": int(r.shots),
         "key_passes": int(r.key_passes)} for r in us_seasons.tail(5).itertuples()]
    tr = base["transfers"]
    transfers = [] if tr.empty else [
        {"date": str(r.transfer_date.date()) if pd.notna(r.transfer_date) else None, "from": r.from_club_name,
         "to": r.to_club_name, "fee_eur": None if pd.isna(r.transfer_fee) else float(r.transfer_fee)}
        for r in tr.itertuples()]
    return {
        "player": {"name": p["name"], "age": age, "nationality": p["country_of_citizenship"],
                   "club": p["current_club_name"],
                   "league": LEAGUE_NAMES_AR.get(base["league_code"], base["league_code"]),
                   "league_strength_vs_premier_league_1_0": round(base["league_strength"], 2),
                   "natural_position": POSITION_NAMES_AR.get(base["pos_group"]),
                   "positions_played_with_games": base["positions_detail"],
                   "foot": p["foot"], "height_cm": p["height_in_cm"], "international_caps": p["international_caps"],
                   "contract_until": None if pd.isna(p["contract_expiration_date"]) else str(p["contract_expiration_date"].date()),
                   "current_injury": base["injury"], "honours": base["trophies"]},
        "age_context": {"age": age, "typical_peak_age_for_position": f"{lo}-{hi}",
                        "years_past_peak": None if age is None else round(max(0.0, age - hi), 1)},
        "team_context_last_league_season": base["team_context"],
        "match_stats_by_season": [{"season": s["season"], "league": s["league"],
                                   "minutes": fm.minutes(s["stats"]),
                                   "sample": "small sample (under 900 minutes)" if fm.minutes(s["stats"]) < 900 else "full",
                                   "stats": _stats_for_ai(s["stats"])}
                                  for s in base["season_stats"]],
        "signature_strengths": signature(base),
        "playing_traits_from_data_0_100": {k: {"label": t["label"], "better_than_pct_of_same_position": t["data_pct"],
                                               "based_on": t["basis"]} for k, t in base["traits_data"].items()},
        "season_average_match_rating": base["season_rating"],
        "recent_match_ratings": recent,
        "xg_by_season_understat": xg,
        "career_by_season": [{k: s[k] for k in ("label", "club", "league", "games", "goals", "assists", "rating", "minutes",
                                                "availability", "market_value_end", "transfer", "tags")}
                             for s in base["seasons"]],
        "transfers": transfers,
        "market_value_by_year": mv_hist,
        "consistency": base["consistency"],
        "trend": base["trend"],
        "valuation_eur": {k: v[k] for k in ("tm_value", "scisports_value", "model_value", "base_value", "fee_low",
                                            "fee_high", "contract_note", "salary_low", "salary_high", "salary_league")},
        "systems_catalogue": [{"name": s["name"], "style": s["style"]} for s in fit.SYSTEMS],
    }


def default_languages(base: dict) -> list[str]:
    leagues = [s["competition_id"] for s in base["seasons"] if s.get("competition_id")]
    return web.auto_languages(base["player"].get("country_of_citizenship"), leagues)


def run_ai(base: dict, target_club: str | None, coach: str | None, progress=None,
           notes: str | None = None, langs: list[str] | None = None) -> dict:
    def step(msg):
        if progress:
            progress(msg)
    p = base["player"]
    step("🌐 جمع آراء المحللين والجمهور والتقارير الفنية والمباريات والإحصاءات...")
    if langs is None:
        langs = default_languages(base)
    items = web.gather(p["name"], p["current_club_name"], base["flags"], target_club, coach, progress=step, langs=langs)
    step("🧠 تحليل موضوعي لكل المصادر مع بيانات المباريات (DeepSeek)...")
    out = ai.analyse(analysis_data(base), web.evidence_text(items), POSITION_NAMES_AR[base["pos_group"]],
                     target_club, coach, notes)
    sources = [{"category": i["category"], "title": i["title"], "url": i["url"], "read": bool(i.get("text"))}
               for i in items if not web.is_internal(i["url"])]
    payload = {"synthesis": out, "sources": sources, "target_club": target_club, "coach": coach, "notes": notes}
    ai.save_cached(base["player_id"], target_club, coach, payload, notes)
    return payload


def style_flat(base: dict) -> dict:
    """Latest full-season (900+ min) deep stats - the fallback style vector for look-alike search."""
    full = [s for s in base["season_stats"] if fm.minutes(s["stats"]) >= 900] or base["season_stats"]
    return full[0]["stats"] if full else {}
