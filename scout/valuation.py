"""Fair-value model (gradient boosting on ~18k Transfermarkt players) + fee & salary ranges."""
from __future__ import annotations

import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

from . import data_tm as tm
from .config import LEAGUE_STRENGTH, MODEL_DIR
from .metrics import league_seasons

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

EUROPE = {"CL", "EL", "UCOL"}
NATIONAL_TOURNAMENTS = {"FIWC", "EURO", "COPA", "AFAC", "AFCN"}
CATEGORICAL = ["pos_group", "sub_position", "league", "foot"]
NUMERIC = ["age", "height_in_cm", "international_caps", "contract_years_left", "league_strength",
           "club_value_ex_self", "min_all", "min_league", "g90", "a90", "ga90_adj", "availability",
           "min_europe", "min_league_prev", "ga90_adj_prev", "min_national_2y"]
FEATURES = CATEGORICAL + NUMERIC

# Annual gross wage as a share of market value, by league (rough market norms, wide on purpose).
WAGE_RATIO = {
    "GB1": (0.10, 0.17), "ES1": (0.08, 0.14), "IT1": (0.08, 0.14), "L1": (0.08, 0.13), "FR1": (0.07, 0.12),
    "SA1": (0.35, 0.90), "MLS1": (0.15, 0.30), "TR1": (0.10, 0.20), "PO1": (0.06, 0.10), "NL1": (0.06, 0.10),
}
DEFAULT_WAGE_RATIO = (0.06, 0.11)


@lru_cache(maxsize=1)
def feature_frame() -> pd.DataFrame:
    p = tm.players()
    last = int(p["last_season"].max())
    ls = league_seasons()
    cur = ls[ls["season"] == last].set_index("player_id")
    prev = ls[ls["season"] == last - 1].set_index("player_id")

    a = tm.appearances()
    a_cur = a[a["season"] == last]
    min_all = a_cur.groupby("player_id")["minutes_played"].sum()
    min_eu = a_cur[a_cur["competition_id"].isin(EUROPE)].groupby("player_id")["minutes_played"].sum()
    nat = a[(a["competition_id"].isin(NATIONAL_TOURNAMENTS)) & (a["date"] >= tm.reference_date() - pd.Timedelta(days=730))]
    min_nat = nat.groupby("player_id")["minutes_played"].sum()

    club_total = p[p["last_season"] == last].groupby("current_club_id")["market_value_in_eur"].sum()

    f = p.set_index("player_id")[["name", "pos_group", "sub_position", "foot", "age", "height_in_cm",
                                  "international_caps", "contract_years_left", "league_strength",
                                  "current_club_domestic_competition_id", "current_club_id",
                                  "market_value_in_eur", "highest_market_value_in_eur", "last_season"]].copy()
    f = f.rename(columns={"current_club_domestic_competition_id": "league"})
    f["international_caps"] = f["international_caps"].fillna(0)
    f["club_value_ex_self"] = np.log1p((f["current_club_id"].map(club_total) - f["market_value_in_eur"].fillna(0)).clip(lower=0))
    f["min_all"] = min_all.reindex(f.index).fillna(0)
    f["min_league"] = cur["minutes"].reindex(f.index).fillna(0)
    for c in ["g90", "a90", "ga90_adj", "availability"]:
        f[c] = cur[c].reindex(f.index)
    f["min_europe"] = min_eu.reindex(f.index).fillna(0)
    f["min_league_prev"] = prev["minutes"].reindex(f.index).fillna(0)
    f["ga90_adj_prev"] = prev["ga90_adj"].reindex(f.index)
    f["min_national_2y"] = min_nat.reindex(f.index).fillna(0)
    for c in CATEGORICAL:
        f[c] = f[c].astype("category")
    return f


def _train_mask(f: pd.DataFrame) -> pd.Series:
    return (f["last_season"] == f["last_season"].max()) & (f["market_value_in_eur"] > 0) & f["league"].isin(LEAGUE_STRENGTH)


def train(save: bool = True) -> dict:
    f = feature_frame()
    d = f[_train_mask(f)]
    X, y = d[FEATURES], np.log(d["market_value_in_eur"])
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, categorical_features="from_dtype", random_state=7)
    pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=7))
    ape = np.abs(np.exp(pred - y) - 1)
    metrics = {"n": int(len(d)), "median_ape": float(np.median(ape)), "within_30pct": float((ape <= 0.30).mean()),
               "within_50pct": float((ape <= 0.50).mean())}
    model.fit(X, y)
    bundle = {"model": model, "metrics": metrics, "reference_date": str(tm.reference_date().date())}
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_DIR / "value_model.joblib")
    return bundle


@lru_cache(maxsize=1)
def load_model() -> dict:
    path = MODEL_DIR / "value_model.joblib"
    if path.exists():
        bundle = joblib.load(path)
        if bundle.get("reference_date") == str(tm.reference_date().date()):
            return bundle
    return train()


@lru_cache(maxsize=1)
def covered_leagues() -> frozenset:
    return frozenset(league_seasons()["competition_id"].unique())


def contract_factor(years_left: float | None) -> tuple[float, float, str]:
    if years_left is None or np.isnan(years_left):
        return 0.85, 1.10, "مدة العقد غير معروفة"
    if years_left <= 0.5:
        return 0.15, 0.45, "عقده ينتهي خلال 6 أشهر - يمكن التفاوض معه مجانًا قريبًا"
    if years_left <= 1:
        return 0.50, 0.80, "سنة واحدة متبقية - موقف تفاوضي قوي للمشتري"
    if years_left <= 2:
        return 0.80, 1.00, "سنتان متبقيتان"
    if years_left <= 3:
        return 0.95, 1.15, "3 سنوات متبقية"
    return 1.05, 1.35, "عقد طويل - النادي البائع في موقف قوي"


def fair_value(player_id: int, scisports_value: float | None = None, target_league: str | None = None) -> dict:
    """Blend (geometric mean) of: our model, Transfermarkt value, and SciSports value (via FotMob, more recent)."""
    bundle = load_model()
    f = feature_frame()
    row = f.loc[[player_id]]
    model_value = float(np.exp(bundle["model"].predict(row[FEATURES])[0]))
    tm_value = float(row["market_value_in_eur"].iloc[0]) if pd.notna(row["market_value_in_eur"].iloc[0]) else None
    # Match-level minutes exist only for the leagues Transfermarkt tracks game-by-game; elsewhere (e.g. Saudi
    # Pro League) the model sees ~0 minutes, so it gets half weight against the market anchors.
    covered = row["league"].iloc[0] in covered_leagues()
    weighted_anchors = [(model_value, 1.0 if covered else 0.5), (tm_value, 1.0), (scisports_value, 1.0)]
    weighted_anchors = [(v, w) for v, w in weighted_anchors if v]
    base = float(np.exp(sum(np.log(v) * w for v, w in weighted_anchors) / sum(w for _, w in weighted_anchors)))

    p = tm.players().set_index("player_id").loc[player_id]
    today = pd.Timestamp.now().normalize()
    years_left = ((p["contract_expiration_date"] - today).days / 365.25) if pd.notna(p["contract_expiration_date"]) else None
    lo, hi, note = contract_factor(years_left)

    league = target_league or row["league"].iloc[0]
    wlo, whi = WAGE_RATIO.get(league, DEFAULT_WAGE_RATIO)
    age = float(row["age"].iloc[0]) if pd.notna(row["age"].iloc[0]) else 26
    age_premium = 1 + 0.07 * max(0.0, age - 28)  # wages stay high while value falls with age
    sal_lo = max(base * wlo * age_premium, 50_000)
    sal_hi = max(base * whi * age_premium, 80_000)

    return {
        "tm_value": tm_value, "scisports_value": scisports_value, "model_value": model_value, "base_value": base,
        "fee_low": base * lo, "fee_high": base * hi, "contract_note": note,
        "contract_years_left": None if years_left is None else round(years_left, 1),
        "salary_low": sal_lo, "salary_high": sal_hi, "salary_league": league,
        "stale": int(row["last_season"].iloc[0]) < int(f["last_season"].max()),
        "model_metrics": bundle["metrics"],
        "drivers": _drivers(row.iloc[0], covered),
        "minutes_covered": covered,
    }


def _drivers(r: pd.Series, covered: bool) -> list[str]:
    out = []
    if pd.notna(r["age"]):
        out.append(f"العمر {r['age']:.0f} سنة")
    if covered:
        out.append(f"دقائق الموسم الأخير {int(r['min_all'])} (منها {int(r['min_europe'])} أوروبيًا)")
    else:
        out.append("دقائق الدوري غير متوفرة في Transfermarkt لهذا الدوري (وزن النموذج مخفّض للنصف)")
    if pd.notna(r["ga90_adj"]):
        out.append(f"مساهمة تهديفية معدّلة {r['ga90_adj']:.2f} / 90")
    out.append(f"قوة الدوري {r['league_strength']:.2f}")
    if r["international_caps"]:
        out.append(f"{int(r['international_caps'])} مباراة دولية")
    return out


def comparables(player_id: int, base_value: float, n: int = 6) -> pd.DataFrame:
    f = feature_frame()
    me = f.loc[player_id]
    c = f[(f.index != player_id) & (f["pos_group"] == me["pos_group"]) & (f["last_season"] == f["last_season"].max())
          & (f["market_value_in_eur"] > 0) & ((f["age"] - me["age"]).abs() <= 2.5)].copy()
    if c.empty:
        return c
    c["dist"] = (np.log(c["market_value_in_eur"]) - np.log(base_value)).abs() \
        + 0.5 * (c["ga90_adj"].fillna(0) - (me["ga90_adj"] if pd.notna(me["ga90_adj"]) else 0)).abs()
    clubs = tm.players().set_index("player_id")["current_club_name"]
    c["club"] = c.index.map(clubs)
    return c.sort_values("dist").head(n)[["name", "club", "league", "age", "market_value_in_eur", "ga90_adj", "min_all"]]
