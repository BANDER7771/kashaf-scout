"""Season tables, per-90 metrics, percentiles and data-driven FC-style ratings."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import data_tm as tm
from . import data_understat as us
from .config import DEFAULT_STRENGTH, LEAGUE_STRENGTH, UNDERSTAT_LEAGUES

MIN_POOL_MINUTES = 900

# percentile -> FC-like 0-99 scale (elite players cluster in the high 80s/low 90s, like the game)
_PCT = [0, 10, 25, 50, 75, 90, 95, 99, 100]
_RATING = [35, 45, 55, 64, 74, 83, 88, 93, 96]


def pct_to_rating(pct: float) -> int:
    return int(round(np.interp(pct, _PCT, _RATING)))


def percentile(value: float, pool: pd.Series) -> float:
    pool = pool.dropna()
    if pool.empty or value is None or np.isnan(value):
        return np.nan
    return float(((pool < value).mean() + 0.5 * (pool == value).mean()) * 100)


@lru_cache(maxsize=1)
def league_seasons() -> pd.DataFrame:
    """Domestic-league totals per player-season (+ team games for availability)."""
    derived = tm.TM_DIR / "league_seasons.parquet"  # prebuilt so servers never hold the full appearances table
    if derived.exists():
        return pd.read_parquet(derived)
    a = tm.appearances()
    a = a[a["is_league"]]
    g = a.groupby(["player_id", "season"])
    out = g.agg(games=("game_id", "count"), minutes=("minutes_played", "sum"), goals=("goals", "sum"),
                assists=("assists", "sum"), yellow=("yellow_cards", "sum"), red=("red_cards", "sum")).reset_index()
    # Main club / competition of the season = where the player logged the most minutes.
    main = (a.groupby(["player_id", "season", "player_club_id", "competition_id"])["minutes_played"].sum()
            .reset_index().sort_values("minutes_played", ascending=False)
            .drop_duplicates(["player_id", "season"]))
    out = out.merge(main[["player_id", "season", "player_club_id", "competition_id"]], on=["player_id", "season"])
    out = out.merge(tm.team_games(), on=["player_club_id", "season", "competition_id"], how="left")
    out["availability"] = (out["minutes"] / (out["team_games"] * 90)).clip(upper=1.0)
    out["strength"] = out["competition_id"].map(LEAGUE_STRENGTH).fillna(DEFAULT_STRENGTH)
    m90 = out["minutes"].where(out["minutes"] > 0) / 90
    out["g90"] = out["goals"] / m90
    out["a90"] = out["assists"] / m90
    out["ga90"] = out["g90"] + out["a90"]
    out["ga90_adj"] = out["ga90"] * out["strength"]
    out["g90_adj"] = out["g90"] * out["strength"]
    out["a90_adj"] = out["a90"] * out["strength"]
    return out


def player_league_seasons(player_id: int) -> pd.DataFrame:
    df = league_seasons()
    df = df[df["player_id"] == player_id].sort_values("season").copy()
    names = tm.clubs().set_index("club_id")["name"]
    df["club"] = df["player_club_id"].map(names).fillna("—")
    df["label"] = [tm.season_label(s, c) for s, c in zip(df["season"], df["competition_id"])]
    return df


@lru_cache(maxsize=8)
def tm_pool(season: int) -> pd.DataFrame:
    df = league_seasons()
    df = df[(df["season"] == season) & (df["minutes"] >= MIN_POOL_MINUTES)]
    pos = tm.players().set_index("player_id")["pos_group"]
    df = df.assign(pos_group=df["player_id"].map(pos))
    return df[df["pos_group"] != "GK"]


@lru_cache(maxsize=8)
def understat_pool(season: int) -> pd.DataFrame:
    frames = [us.league_players(lg, season) for lg in UNDERSTAT_LEAGUES.values()]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["time"] >= MIN_POOL_MINUTES) & (~df["position"].str.startswith("GK"))].copy()
    return add_understat_rates(df)


def add_understat_rates(df: pd.DataFrame) -> pd.DataFrame:
    m90 = df["time"] / 90
    df["npxG90"] = df["npxG"] / m90
    df["npg90"] = df["npg"] / m90
    df["xA90"] = df["xA"] / m90
    df["kp90"] = df["key_passes"] / m90
    df["shots90"] = df["shots"] / m90
    df["xGChain90"] = df["xGChain"] / m90
    df["xGBuildup90"] = df["xGBuildup"] / m90
    # Finishing over/under expected per shot, shrunk towards 0 for small samples.
    df["fin_over"] = (df["npg"] - df["npxG"]) / (df["shots"] + 20)
    return df


def pick_rating_season(seasons: pd.DataFrame) -> pd.Series | None:
    """Latest season with meaningful minutes (falls back to the latest season at all)."""
    if seasons.empty:
        return None
    ok = seasons[seasons["minutes"] >= 600]
    return (ok if not ok.empty else seasons).iloc[-1]


def pick_understat_row(us_seasons: pd.DataFrame, season: int) -> pd.Series | None:
    if us_seasons.empty:
        return None
    cand = us_seasons[(us_seasons["time"] >= 600) & (us_seasons["season"] >= season - 2)]
    if cand.empty:
        return None
    same = cand[cand["season"] == season]
    return (same if not same.empty else cand).iloc[-1]


def data_ratings(pos_group: str, season_row: pd.Series | None, us_seasons: pd.DataFrame) -> dict:
    """FC-style attribute ratings that can be derived from free data.

    Absolute scale: compared against *all* outfield regulars (like FC attributes),
    so a centre-back's finishing is low even if he is a good centre-back.
    Returns {attr: {"value": int, "detail": str}}.
    """
    out: dict[str, dict] = {}
    if pos_group == "GK" or season_row is None or season_row["minutes"] < 600:
        return out
    pool = tm_pool(int(season_row["season"]))
    us_row = pick_understat_row(us_seasons, int(season_row["season"]))
    upool = understat_pool(int(us_row["season"])) if us_row is not None else pd.DataFrame()

    if us_row is not None and not upool.empty:
        recent = add_understat_rates(us_seasons[us_seasons["season"] >= us_row["season"] - 2].sum(numeric_only=True).to_frame().T)
        p_npg = percentile(us_row["npg"] / us_row["time"] * 90, upool["npg90"])
        p_npxg = percentile(us_row["npxG"] / us_row["time"] * 90, upool["npxG90"])
        p_fin = percentile(float(recent["fin_over"].iloc[0]), upool["fin_over"])
        p_xa = percentile(us_row["xA"] / us_row["time"] * 90, upool["xA90"])
        p_kp = percentile(us_row["key_passes"] / us_row["time"] * 90, upool["kp90"])
        p_bu = percentile(us_row["xGBuildup"] / us_row["time"] * 90, upool["xGBuildup90"])
        out["finishing"] = {"value": pct_to_rating(0.5 * p_npg + 0.3 * p_npxg + 0.2 * p_fin),
                            "detail": f"أهداف (بدون ركلات جزاء)/90 أعلى من {p_npg:.0f}% · xG/90 أعلى من {p_npxg:.0f}% · الإنهاء مقارنة بـ xG أعلى من {p_fin:.0f}% من اللاعبين"}
        out["positioning"] = {"value": pct_to_rating(p_npxg),
                              "detail": f"يصل لفرص خطرة (npxG/90) أكثر من {p_npxg:.0f}% من لاعبي الدوريات الكبرى"}
        out["vision"] = {"value": pct_to_rating(0.55 * p_xa + 0.45 * p_kp),
                         "detail": f"xA/90 أعلى من {p_xa:.0f}% · تمريرات مفتاحية/90 أعلى من {p_kp:.0f}%"}
        out["passing"] = {"value": pct_to_rating(p_bu),
                          "detail": f"المشاركة في البناء (xGBuildup/90) أعلى من {p_bu:.0f}%"}
        if recent["shots"].iloc[0] >= 25:
            out["composure"] = {"value": pct_to_rating(p_fin),
                                "detail": f"تحويل الفرص مقارنة بجودتها (آخر 3 مواسم، {int(recent['shots'].iloc[0])} تسديدة)"}
    elif not pool.empty:
        p_g = percentile(season_row["g90_adj"], pool["g90_adj"])
        p_a = percentile(season_row["a90_adj"], pool["a90_adj"])
        out["finishing"] = {"value": pct_to_rating(p_g),
                            "detail": f"أهداف/90 (معدّلة حسب قوة الدوري) أعلى من {p_g:.0f}% من لاعبي 31 دوري"}
        out["vision"] = {"value": pct_to_rating(p_a),
                         "detail": f"تمريرات حاسمة/90 (معدّلة) أعلى من {p_a:.0f}%"}
    return out


def peer_metrics(pos_group: str, season_row: pd.Series | None, us_seasons: pd.DataFrame) -> list[dict]:
    """Per-90 metrics with percentiles vs. the same position group (for the comparison table)."""
    rows: list[dict] = []
    if season_row is None or season_row["minutes"] < 600:
        return rows
    pool = tm_pool(int(season_row["season"]))
    peers = pool[pool["pos_group"] == pos_group]
    for key, label in [("g90", "أهداف / 90"), ("a90", "صناعة / 90"), ("ga90_adj", "مساهمة تهديفية معدّلة / 90"),
                       ("availability", "نسبة المشاركة من دقائق الفريق")]:
        rows.append({"metric": label, "value": season_row[key], "pct": percentile(season_row[key], peers[key]),
                     "source": "Transfermarkt"})
    us_row = pick_understat_row(us_seasons, int(season_row["season"]))
    if us_row is not None:
        upool = understat_pool(int(us_row["season"]))
        if not upool.empty:
            letter = {"ST": "F", "W": "F", "AM": "M", "CM": "M", "DM": "M", "FB": "D", "CB": "D"}.get(pos_group, "M")
            upeers = upool[upool["position"].str.startswith(letter)]
            r = add_understat_rates(pd.DataFrame([us_row.drop(labels=["position", "team"], errors="ignore").astype(float)]))
            for key, label in [("npxG90", "xG بدون جزاء / 90"), ("xA90", "xA / 90"), ("shots90", "تسديدات / 90"),
                               ("kp90", "تمريرات مفتاحية / 90"), ("xGChain90", "xGChain / 90"),
                               ("xGBuildup90", "xGBuildup / 90")]:
                v = float(r[key].iloc[0])
                rows.append({"metric": label, "value": v, "pct": percentile(v, upeers[key]), "source": "Understat"})
    return rows


# ---------------------------------------------------------------- FotMob (Opta) based ratings

# attribute -> {fotmob stat id: weight}. FotMob percentiles are per-90, vs. same position in the same league,
# already oriented so that higher = better (e.g. fewer times dribbled past -> higher percentile).
FOTMOB_ATTR = {
    "finishing": {"goals": .40, "non_penalty_xg": .30, "ShotsOnTarget": .30},
    "positioning": {"non_penalty_xg": .50, "touches_opp_box": .50},
    "vision": {"chances_created": .35, "expected_assists": .35, "big_chance_created_team_title": .30},
    "passing": {"successful_passes": .40, "successful_passes_accuracy": .35, "long_balls_accurate": .25},
    "dribbling": {"dribbles_succeeded": .50, "won_contest_subtitle": .20, "dispossessed": .30},
    # Volume stats (tackles, interceptions) drop in possession-dominant teams, so success-rate and
    # "not beaten" signals carry more weight than raw counts.
    "defensive_awareness": {"dribbled_past": .35, "interceptions": .25, "clearances": .20, "recoveries": .20},
    "tackling": {"duel_won_percent": .45, "matchstats.headers.tackles": .35, "defensive_actions": .20},
    "aerial": {"aerials_won": .50, "aerials_won_percent": .50},
    "work_rate": {"poss_won_att_3rd_team_title": .40, "recoveries": .30, "defensive_actions": .30},
    "composure": {"dispossessed": .50, "successful_passes_accuracy": .50},
    "strength": {"duel_won_percent": .60, "aerials_won_percent": .40},
}
# Percentiles are position-relative, so only use them where the comparison is meaningful for the role
# (a centre-back in the 90th percentile for goals among centre-backs is not a 90-rated finisher).
DATA_ATTRS_BY_POS = {
    "ST": ["finishing", "positioning", "vision", "passing", "dribbling", "aerial", "work_rate", "composure", "strength"],
    "W": ["finishing", "positioning", "vision", "passing", "dribbling", "work_rate", "composure"],
    "AM": ["finishing", "positioning", "vision", "passing", "dribbling", "work_rate", "composure"],
    "CM": ["vision", "passing", "dribbling", "defensive_awareness", "tackling", "work_rate", "composure"],
    "DM": ["passing", "defensive_awareness", "tackling", "aerial", "work_rate", "composure", "strength"],
    "FB": ["vision", "passing", "dribbling", "defensive_awareness", "tackling", "work_rate"],
    "CB": ["passing", "defensive_awareness", "tackling", "aerial", "composure", "strength"],
    "GK": [],
}
# Secondary skills for a role sit lower on FC's absolute scale than the same percentile in a core skill.
POS_OFFSET = {
    "ST": {"passing": -6, "vision": -4, "work_rate": -3},
    "W": {"passing": -3, "work_rate": -3},
    "AM": {"work_rate": -3},
    "CM": {"defensive_awareness": -4, "tackling": -4, "dribbling": -3},
    "DM": {"aerial": -4},
    "FB": {"vision": -5, "dribbling": -4},
    "CB": {"passing": -6},
}
FOTMOB_LEAGUE_CODE = {
    "Premier League": "GB1", "LaLiga": "ES1", "Serie A": "IT1", "Bundesliga": "L1", "Ligue 1": "FR1",
    "Saudi Pro League": "SA1", "Eredivisie": "NL1", "Liga Portugal": "PO1", "Liga Portugal Betclic": "PO1",
    "Super Lig": "TR1", "Süper Lig": "TR1", "MLS": "MLS1", "Major League Soccer": "MLS1",
    "Jupiler Pro League": "BE1", "First Division A": "BE1", "Scottish Premiership": "SC1",
    "Premiership": "SC1", "Brasileirão Série A": "BRA1", "Serie A Betano": "BRA1", "Liga MX": "MEX1",
    "Liga Profesional": "ARG1", "Superliga": "DK1", "Eliteserien": "NO1", "Allsvenskan": "SE1",
    "J. League": "JAP1", "J1 League": "JAP1", "K League 1": "RSK1", "A-League": "AUS1", "Ekstraklasa": "PL1",
    "Super League 1": "GR1", "Swiss Super League": "C1", "Austrian Bundesliga": "A1", "Czech First League": "TS1",
    "Championship": "GB2", "Qatar Stars League": "QSL", "Pro League": "UAE1", "UAE Pro League": "UAE1",
    "HNL": "KR1", "Premier Liga": "RU1", "Russian Premier League": "RU1", "Ukrainian Premier League": "UKR1",
}


def fotmob_league_strength(league_name: str | None, fallback_code: str | None) -> tuple[float, str | None]:
    code = FOTMOB_LEAGUE_CODE.get(league_name or "") or fallback_code
    return LEAGUE_STRENGTH.get(code, DEFAULT_STRENGTH), code


def _wpct(flat: dict, weights: dict) -> tuple[float | None, list[str]]:
    have = {k: w for k, w in weights.items() if flat.get(k, {}).get("pct") is not None}
    if sum(have.values()) < 0.5:
        return None, []
    pct = sum(flat[k]["pct"] * w for k, w in have.items()) / sum(have.values())
    used = [f"{stat_label(k, flat[k]['title'])}: أفضل من {flat[k]['pct']:.0f}%" for k in have]
    return pct, used


def fotmob_ratings(pos_group: str, flat: dict, strength: float, season: str, league: str) -> dict:
    out = {}
    league_adj = -(1 - strength) * 15  # e.g. Saudi Pro League ~ -4, Premier League 0
    for attr in DATA_ATTRS_BY_POS.get(pos_group, []):
        pct, used = _wpct(flat, FOTMOB_ATTR[attr])
        if pct is None:
            continue
        val = pct_to_rating(pct) + POS_OFFSET.get(pos_group, {}).get(attr, 0) + league_adj
        out[attr] = {"value": int(np.clip(round(val), 25, 95)),
                     "detail": f"{league} {season} - مقارنة بلاعبي نفس المركز: " + "، ".join(used)
                               + (f" (تعديل قوة الدوري {league_adj:+.0f})" if league_adj <= -1 else "")}
    return out


FOTMOB_PEER_STATS = {
    "attack": ["goals", "non_penalty_xg", "expected_assists", "chances_created", "dribbles_succeeded",
               "touches_opp_box", "successful_passes_accuracy", "poss_won_att_3rd_team_title"],
    "mid": ["expected_assists", "chances_created", "successful_passes", "successful_passes_accuracy",
            "long_balls_accurate", "dribbles_succeeded", "matchstats.headers.tackles", "interceptions", "recoveries"],
    "def": ["matchstats.headers.tackles", "interceptions", "clearances", "recoveries", "aerials_won_percent",
            "duel_won_percent", "dribbled_past", "successful_passes_accuracy", "long_balls_accurate"],
}


def fotmob_peer_metrics(pos_group: str, flat: dict) -> list[dict]:
    kind = "attack" if pos_group in ("ST", "W", "AM") else "mid" if pos_group in ("CM", "DM") else "def"
    rows = []
    for key in ["rating"] + FOTMOB_PEER_STATS[kind]:
        s = flat.get(key)
        if not s or s.get("per90") is None:
            continue
        rows.append({"metric": stat_label(key, s["title"]), "value": float(s["per90"]), "pct": s.get("pct"),
                     "source": "FotMob (Opta)"})
    return rows


# ---------------------------------------------------------------- playing traits (for the summary)

TRAITS = {
    "physical_effort": "المجهود البدني والركض",
    "ball_carrying": "التقدم بالكرة",
    "press_resistance": "الاحتفاظ بالكرة تحت الضغط",
    "progression": "الكرات الطويلة والتمرير التقدمي",
    "chance_creation": "صناعة الفرص والرؤية",
    "end_product": "الحسم واللمسة الأخيرة",
    "tactical_intelligence": "الذكاء التكتيكي والتمركز",
    "open_space": "اللعب في المساحات المفتوحة والتحولات",
    "tight_space": "اللعب في المساحات الضيقة",
    "pressing": "الضغط على الخصم",
    "duels": "الالتحامات والكرات الهوائية",
    "defensive_discipline": "الانضباط الدفاعي",
    "consistency": "ثبات المستوى",
}
_ATTACKERS = ("ST", "W", "AM")
# Volume-based counts punish players who are always on the ball (more touches -> more times dispossessed), so
# ball-retention traits lean on success rates instead.
TRAIT_STATS = {
    "physical_effort": {"recoveries": .35, "defensive_actions": .35, "duel_won": .30},
    "ball_carrying": {"dribbles_succeeded": .45, "won_contest_subtitle": .20, "touches": .20, "fouls_won": .15},
    "press_resistance": {"successful_passes_accuracy": .35, "won_contest_subtitle": .30, "touches": .15, "fouls_won": .20},
    "progression": {"long_balls_accurate": .40, "successful_passes": .30, "chances_created": .30},
    "chance_creation": {"chances_created": .40, "expected_assists": .35, "big_chance_created_team_title": .25},
    "end_product": {"goals": .35, "non_penalty_xg": .25, "expected_assists": .20, "big_chance_created_team_title": .20},
    "open_space": {"dribbles_succeeded": .40, "big_chance_created_team_title": .20, "non_penalty_xg": .20, "fouls_won": .20},
    "tight_space": {"won_contest_subtitle": .35, "successful_passes_accuracy": .25, "dispossessed": .20, "touches_opp_box": .20},
    "pressing": {"poss_won_att_3rd_team_title": .60, "defensive_actions": .20, "recoveries": .20},
    "duels": {"duel_won_percent": .40, "aerials_won_percent": .30, "aerials_won": .30},
    # "Rarely dribbled past" alone flatters players who seldom defend, so volume of defending is part of it.
    "defensive_discipline": {"dribbled_past": .40, "defensive_actions": .30, "fouls": .15, "yellow_cards": .15},
}
# How much the measured baseline counts when merged with the AI's evidence-based judgement. Running volume and
# speed are not measured at all in free data, so reports carry most of the weight there.
TRAIT_DATA_WEIGHT = {"physical_effort": .30, "open_space": .35, "tactical_intelligence": .40, "consistency": .70}


def trait_signals(pos_group: str, flat: dict, consistency_score: float | None) -> dict:
    """Data side of each playing trait: weighted percentile vs. same position in the same league (0-100)."""
    stats = dict(TRAIT_STATS)
    stats["tactical_intelligence"] = ({"non_penalty_xg": .50, "touches_opp_box": .30, "expected_assists": .20}
                                      if pos_group in _ATTACKERS else
                                      {"interceptions": .40, "dribbled_past": .30, "recoveries": .30})
    out = {}
    for key, weights in stats.items():
        pct, used = _wpct(flat or {}, weights)
        if pct is not None:
            out[key] = {"label": TRAITS[key], "data_pct": round(pct), "basis": "، ".join(used)}
    if consistency_score is not None:
        out["consistency"] = {"label": TRAITS["consistency"], "data_pct": round(consistency_score),
                              "basis": "درجة الثبات المحسوبة من تقييمات المباريات والمواسم"}
    return out


STAT_AR = {
    "goals": "الأهداف", "expected_goals": "xG", "expected_goals_on_target": "xGOT (جودة التسديد على المرمى)",
    "goals_subtitle": "أهداف من ركلات جزاء", "non_penalty_xg": "xG بدون ركلات الجزاء", "shots": "التسديدات",
    "ShotsOnTarget": "التسديدات على المرمى", "headed_shots": "التسديدات بالرأس", "assists": "التمريرات الحاسمة",
    "expected_assists": "xA", "successful_passes": "التمريرات الصحيحة", "successful_passes_accuracy": "دقة التمرير",
    "long_balls_accurate": "الكرات الطويلة الصحيحة", "long_ball_succeeeded_accuracy": "دقة الكرات الطويلة",
    "chances_created": "الفرص المصنوعة", "big_chance_created_team_title": "الفرص الكبيرة المصنوعة",
    "crosses_succeeeded": "العرضيات الناجحة", "crosses_succeeeded_accuracy": "دقة العرضيات",
    "dribbles_succeeded": "المراوغات الناجحة", "won_contest_subtitle": "نسبة نجاح المراوغة",
    "duel_won": "الالتحامات المكسوبة", "duel_won_percent": "نسبة الفوز بالالتحامات",
    "aerials_won": "الكرات الهوائية المكسوبة", "aerials_won_percent": "نسبة الفوز بالكرات الهوائية",
    "touches": "اللمسات", "touches_opp_box": "اللمسات في منطقة جزاء الخصم", "dispossessed": "مرات فقدان الكرة",
    "fouls_won": "الأخطاء المكتسبة", "penalty_won_title": "ركلات الجزاء المكتسبة",
    "defensive_actions": "الأفعال الدفاعية", "matchstats.headers.tackles": "الافتكاكات",
    "interceptions": "الاعتراضات", "fouls": "الأخطاء المرتكبة", "recoveries": "استعادة الكرة",
    "poss_won_att_3rd_team_title": "استعادة الكرة في الثلث الأخير", "dribbled_past": "مرات تجاوزه بالمراوغة",
    "clearances": "الإبعادات", "clean_sheet_team_title": "مباريات بشباك نظيفة",
    "goals_conceded_while_on_pitch": "أهداف استقبلها الفريق بوجوده",
    "expected_goals_against_while_on_pitch": "xG ضد الفريق بوجوده", "yellow_cards": "البطاقات الصفراء",
    "red_cards": "البطاقات الحمراء", "rating": "متوسط التقييم", "matches_uppercase": "المباريات",
    "player_started_matches": "مباريات أساسيًا", "minutes_played": "الدقائق",
}
def stat_label(key: str, fallback: str | None = None) -> str:
    return STAT_AR.get(key, fallback or key)


def describe_stat(key: str, s: dict) -> str:
    """One unambiguous Arabic line per stat: the value itself, then the percentile rank as a separate idea."""
    is_pct = s.get("format") == "percent" or "accuracy" in key or "percent" in key or key == "won_contest_subtitle"
    unit = "%" if is_pct else ""
    parts = [f"القيمة {s.get('value')}{unit}"]
    if s.get("per90") is not None and unit == "" and key not in ("rating", "matches_uppercase", "player_started_matches",
                                                                 "minutes_played"):
        parts.append(f"لكل 90 دقيقة {float(s['per90']):.2f}")
    if s.get("pct") is not None:
        # FotMob ranks are already oriented so that higher = better (fewer times dispossessed ranks higher).
        parts.append(f"ترتيبه: أفضل من {float(s['pct']):.0f}% من لاعبي مركزه في الدوري")
    return " · ".join(parts)
