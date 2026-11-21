"""Career timeline (FotMob + Transfermarkt merged), turning points, consistency and trend."""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import data_tm as tm
from .config import CALENDAR_LEAGUES
from .data_tm import normalize_name
from .metrics import FOTMOB_LEAGUE_CODE

ATTACKING = {"ST", "W", "AM", "CM"}
SEASON_COLS = ["season", "label", "club", "league", "competition_id", "games", "goals", "assists", "rating",
               "minutes", "availability", "ga90", "ga90_adj"]


def _clip(x: float) -> float:
    return float(np.clip(x, 0, 100))


def _fm_label(label: str) -> str:
    return f"{label[:4]}/{label[-2:]}" if "/" in label else label


def unified_seasons(tm_seasons: pd.DataFrame, fm_seasons: pd.DataFrame) -> pd.DataFrame:
    """One row per season-club in a domestic league. FotMob gives every league + average rating;
    Transfermarkt adds minutes / availability for the 14 European leagues it tracks match-by-match."""
    rows = []
    used_tm = set()
    tm_by_season = {s: g for s, g in tm_seasons.groupby("season")} if not tm_seasons.empty else {}
    for f in fm_seasons.itertuples() if not fm_seasons.empty else []:
        row = {"season": f.season, "label": _fm_label(f.label), "club": f.club, "league": f.league,
               "competition_id": FOTMOB_LEAGUE_CODE.get(f.league), "games": f.games, "goals": f.goals,
               "assists": f.assists, "rating": f.rating, "minutes": None, "availability": None,
               "ga90": None, "ga90_adj": None}
        cands = tm_by_season.get(f.season)
        if cands is not None:
            scored = [(fuzz.partial_ratio(normalize_name(f.club), normalize_name(c.club)), c) for c in cands.itertuples()]
            score, best = max(scored, key=lambda t: t[0])
            if score >= 60 or (len(cands) == 1 and (fm_seasons["season"] == f.season).sum() == 1):
                used_tm.add(best.Index)
                row.update(minutes=int(best.minutes), availability=best.availability, ga90=best.ga90,
                           ga90_adj=best.ga90_adj, competition_id=best.competition_id)
        rows.append(row)
    for t in tm_seasons.itertuples() if not tm_seasons.empty else []:
        if t.Index in used_tm:
            continue
        rows.append({"season": t.season, "label": t.label, "club": t.club, "league": t.competition_id,
                     "competition_id": t.competition_id, "games": t.games, "goals": t.goals, "assists": t.assists,
                     "rating": None, "minutes": int(t.minutes), "availability": t.availability,
                     "ga90": t.ga90, "ga90_adj": t.ga90_adj})
    df = pd.DataFrame(rows, columns=SEASON_COLS)
    for c in ["rating", "minutes", "availability", "ga90", "ga90_adj"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["season", "games"]).reset_index(drop=True)


def current_season() -> int:
    now = pd.Timestamp.now()
    return now.year if now.month >= 7 else now.year - 1


def market_values(player_id: int, fm_values: pd.DataFrame) -> pd.DataFrame:
    v = tm.valuations_for(player_id).sort_values("date")[["date", "market_value_in_eur", "current_club_name"]]
    v = v.assign(source="Transfermarkt")
    if not fm_values.empty:
        newer = fm_values[fm_values["date"] > (v["date"].max() if not v.empty else pd.Timestamp.min)]
        v = pd.concat([v, newer.assign(source="SciSports (FotMob)")], ignore_index=True)
    return v


def player_transfers(player_id: int) -> pd.DataFrame:
    t = tm.transfers_for(player_id)
    t = t[t["transfer_date"] <= pd.Timestamp.now() + pd.Timedelta(days=30)]
    return t.sort_values("transfer_date")[["transfer_date", "transfer_season", "from_club_name", "to_club_name",
                                           "transfer_fee", "market_value_in_eur"]]


def _mv_at(mv: pd.DataFrame, date: pd.Timestamp) -> float:
    before = mv[mv["date"] <= date]
    return float(before["market_value_in_eur"].iloc[-1]) if not before.empty else np.nan


def _season_end(season: int, competition: str | None) -> pd.Timestamp:
    return pd.Timestamp(f"{season}-12-31") if competition in CALENDAR_LEAGUES else pd.Timestamp(f"{season + 1}-06-30")


def _none(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def timeline(pos_group: str, seasons: pd.DataFrame, mv: pd.DataFrame, tr: pd.DataFrame) -> list[dict]:
    rows = []
    for s in seasons.itertuples():
        end = _season_end(int(s.season), s.competition_id)
        start = end - pd.Timedelta(days=365)
        moved = tr[(tr["transfer_date"] > start - pd.Timedelta(days=60)) & (tr["transfer_date"] <= end)] if not tr.empty else tr
        rows.append({
            "season": int(s.season), "label": s.label, "club": s.club, "league": s.league,
            "competition_id": s.competition_id, "games": int(s.games), "goals": int(s.goals), "assists": int(s.assists),
            "rating": _none(s.rating), "minutes": _none(s.minutes),
            "ga90_adj": None if _none(s.ga90_adj) is None else round(float(s.ga90_adj), 2),
            "availability": None if _none(s.availability) is None else round(float(s.availability), 2),
            "market_value_end": _none(_mv_at(mv, end)) if not mv.empty else None,
            "transfer": "، ".join(dict.fromkeys(f"{r.from_club_name} ← {r.to_club_name}" for r in moved.itertuples())) or None,
            "ongoing": int(s.season) >= current_season() and s.competition_id not in CALENDAR_LEAGUES,
            "tags": [],
        })
    _tag_phases(rows, pos_group)
    return rows


def _tag_phases(rows: list[dict], pos_group: str) -> None:
    rated = [r for r in rows if r["rating"] and r["games"] >= 10]
    if rated:
        max(rated, key=lambda r: r["rating"])["tags"].append("ذروة التقييم")
    regular = [r for r in rows if (r["minutes"] or 0) >= 900 and r["ga90_adj"] is not None]
    if pos_group in ATTACKING and regular:
        max(regular, key=lambda r: r["ga90_adj"])["tags"].append("ذروة الإنتاج")
    with_mv = [r for r in rows if r["market_value_end"]]
    if with_mv:
        max(with_mv, key=lambda r: r["market_value_end"])["tags"].append("ذروة القيمة")
    for prev, cur in zip(rows, rows[1:]):
        if cur["season"] == prev["season"]:
            continue  # mid-season move: compare across seasons only
        if cur["transfer"]:
            cur["tags"].append("انتقال")
        if prev["rating"] and cur["rating"] and prev["games"] >= 10 and cur["games"] >= 10:
            d = cur["rating"] - prev["rating"]
            if d <= -0.30:
                cur["tags"].append(f"تراجع التقييم {d:+.2f}")
            elif d >= 0.30:
                cur["tags"].append(f"تحسّن التقييم {d:+.2f}")
        if pos_group in ATTACKING and (prev["minutes"] or 0) >= 900 and (cur["minutes"] or 0) >= 900 \
                and prev["ga90_adj"] and cur["ga90_adj"] is not None:
            ch = cur["ga90_adj"] / prev["ga90_adj"] - 1
            if ch <= -0.30:
                cur["tags"].append(f"تراجع الإنتاج {ch:.0%}")
            elif ch >= 0.30:
                cur["tags"].append(f"تحسّن الإنتاج +{ch:.0%}")
        pa, ca = prev["availability"], cur["availability"]
        if pa is not None and ca is not None:
            if pa >= 0.6 and ca < 0.35:
                cur["tags"].append("مشاركة قليلة (إصابة أو دكة؟)")
            elif pa < 0.35 and ca >= 0.6:
                cur["tags"].append("عودة للأساسية")
        elif prev["games"] >= 20 and cur["games"] < 10 and not cur["ongoing"]:
            cur["tags"].append("مباريات قليلة (إصابة أو دكة؟)")
        pm, cm = prev["market_value_end"], cur["market_value_end"]
        if pm and cm and not cur["ongoing"]:
            if cm / pm - 1 <= -0.30:
                cur["tags"].append(f"هبوط القيمة {cm / pm - 1:.0%}")
            elif cm / pm - 1 >= 0.50:
                cur["tags"].append(f"قفزة في القيمة +{cm / pm - 1:.0%}")


def consistency(player_id: int, pos_group: str, seasons: pd.DataFrame, us_matches: pd.DataFrame,
                fm_matches: pd.DataFrame, mv: pd.DataFrame) -> dict:
    parts: dict[str, float] = {}
    notes: list[str] = []

    if not fm_matches.empty and len(fm_matches) >= 12:
        r = fm_matches.tail(40)["rating"]
        std = r.std()
        # Match ratings swing more for attackers (goals move the rating), so the baseline is position-aware.
        base = 0.85 if pos_group in ("ST", "W", "AM") else 0.70 if pos_group in ("CM", "DM") else 0.55
        parts["تقييمات المباريات"] = _clip(100 - (std - base) * 100)
        notes.append(f"آخر {len(r)} مباراة: متوسط التقييم {r.mean():.2f}، الانحراف المعياري {std:.2f}، "
                     f"مباريات ضعيفة (أقل من 6.5) {(r < 6.5).mean():.0%}")

    rated = seasons[(seasons["games"] >= 10) & seasons["rating"].notna()].drop_duplicates("season", keep="last").tail(4)
    if len(rated) >= 3:
        std = rated["rating"].std()
        parts["بين المواسم (التقييم)"] = _clip(100 - (std - 0.05) * 200)
        notes.append(f"تقييم المواسم الأخيرة: {' ← '.join(f'{x:.2f}' for x in rated['rating'])} (انحراف {std:.2f})")

    recent = seasons.drop_duplicates("season", keep="last").tail(5)
    regular = recent[(recent["minutes"].fillna(0) >= 900) & recent["ga90_adj"].notna()]
    if pos_group in ATTACKING and len(regular) >= 3 and regular["ga90_adj"].astype(float).mean() > 0:
        g = regular["ga90_adj"].astype(float)
        cv = g.std() / g.mean()
        parts["الإنتاج التهديفي بين المواسم"] = _clip(100 - (cv - 0.1) * 150)
        notes.append(f"معامل تباين المساهمة التهديفية (آخر {len(regular)} مواسم) = {cv:.2f}")

    if pos_group in ATTACKING and not us_matches.empty:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=2 * 365)
        m = us_matches[(us_matches["date"] >= cutoff) & (us_matches["time"] >= 45)]
        if len(m) >= 16:
            val = ((m["npxG"] + m["xA"]).rolling(8).sum() / m["time"].rolling(8).sum() * 90).dropna()
            if val.mean() > 0:
                cv = val.std() / val.mean()
                parts["xG+xA مباراة بمباراة"] = _clip(100 - (cv - 0.2) * 120)
                notes.append(f"تذبذب xG+xA على نوافذ 8 مباريات ({len(m)} مباراة) = {cv:.2f}")

    avail = recent["availability"].dropna().astype(float)
    if len(avail) >= 3:
        parts["انتظام المشاركة"] = _clip(100 - avail.std() * 200)
        notes.append(f"نسبة المشاركة من دقائق الفريق: متوسط {avail.mean():.0%}، انحراف {avail.std():.2f}")

    if not parts and len(mv) >= 4:
        mv4 = mv[mv["date"] >= pd.Timestamp.now() - pd.Timedelta(days=4 * 365)]
        if len(mv4) >= 4:
            vol = np.log(mv4["market_value_in_eur"].clip(lower=1)).diff().std()
            parts["استقرار القيمة"] = _clip(100 - vol * 150)
            notes.append(f"تذبذب القيمة السوقية = {vol:.2f}")

    if not parts:
        return {"score": None, "label": "بيانات غير كافية", "parts": {}, "notes": notes}
    score = float(np.mean(list(parts.values())))
    label = "ثابت جدًا" if score >= 75 else "ثابت" if score >= 60 else "متذبذب نسبيًا" if score >= 45 else "متذبذب"
    return {"score": round(score), "label": label, "parts": {k: round(v) for k, v in parts.items()}, "notes": notes}


def trend(seasons: pd.DataFrame, mv: pd.DataFrame) -> dict:
    out = {}
    rated = seasons[(seasons["games"] >= 10) & seasons["rating"].notna()].drop_duplicates("season", keep="last").tail(3)
    if len(rated) >= 2:
        slope = float(np.polyfit(range(len(rated)), rated["rating"].astype(float), 1)[0])
        out["performance"] = "في صعود" if slope > 0.10 else "في هبوط" if slope < -0.10 else "مستقر"
        out["rating_slope_per_season"] = round(slope, 2)
    if len(mv) >= 2:
        now = mv["market_value_in_eur"].iloc[-1]
        past = mv[mv["date"] <= mv["date"].iloc[-1] - pd.Timedelta(days=540)]
        if not past.empty and past["market_value_in_eur"].iloc[-1] > 0:
            ch = now / past["market_value_in_eur"].iloc[-1] - 1
            out["value"] = "في صعود" if ch > 0.2 else "في هبوط" if ch < -0.2 else "مستقرة"
            out["value_change_18m"] = round(float(ch), 2)
        peak = mv.loc[mv["market_value_in_eur"].idxmax()]
        out["peak_value"] = float(peak["market_value_in_eur"])
        out["peak_date"] = peak["date"].strftime("%Y-%m")
    return out
