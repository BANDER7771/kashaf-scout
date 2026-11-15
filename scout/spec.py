"""Search by specification: natural-language request -> structured spec (DeepSeek) -> ranked players with
expected fee, free-agent status and salary."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import ai, pool, valuation
from .config import LEAGUE_NAMES_AR, POSITION_NAMES_AR, POOL_LEAGUES
from .valuation import WAGE_RATIO, DEFAULT_WAGE_RATIO, contract_factor

PARSE_SYSTEM = """You turn a football recruiter's request (usually Arabic) into a search specification. \
Return JSON exactly matching the template. Use null for anything not mentioned. Map playing qualities to \
style_weights on a 0-1 scale (1 = essential, 0.5 = useful) using ONLY these feature keys: {features}. Use \
negative weights (e.g. -0.3) only for qualities the user explicitly does NOT want. Physical tracking features \
(dist90, sprints90, topspeed) exist only for Premier League players - use them only if the user asks for \
running, intensity or speed. Values in euros. Ages in years."""

PARSE_TEMPLATE = {
    "positions": [f"subset of {list(POSITION_NAMES_AR)} (GK, CB, FB, DM, CM, AM, W, ST)"],
    "age_min": "int or null", "age_max": "int or null",
    "max_value_eur": "number or null", "min_value_eur": "number or null",
    "scope": "all | arab | europe",
    "leagues": [f"optional subset of {[c for c, _ in POOL_LEAGUES.values()]}"],
    "foot": "left | right | null", "min_height": "int cm or null", "nationality": "English country name or null",
    "contract_expiring": "true if the user wants free agents / expiring contracts",
    "style_weights": {"feature_key": "weight"},
    "quality_weight": "0-1, how much current level matters vs style (default 0.35)",
    "understood_ar": "Arabic, one sentence restating what you understood",
}


def parse(text: str) -> dict:
    feats = {k: pool.FEATURE_AR[k] for k in pool.STYLE_FEATURES + pool.PHYS_FEATURES}
    system = PARSE_SYSTEM.replace("{features}", json.dumps(feats, ensure_ascii=False))
    out, _ = ai.chat(system, f"TEMPLATE:\n{json.dumps(PARSE_TEMPLATE, ensure_ascii=False)}\n\nREQUEST:\n{text}",
                     max_tokens=1500, temperature=0.1)
    out["positions"] = [p for p in out.get("positions") or [] if p in POSITION_NAMES_AR]
    out["scope"] = out.get("scope") if out.get("scope") in ("all", "arab", "europe") else "all"
    codes = {c for c, _ in POOL_LEAGUES.values()}
    out["leagues"] = [lg for lg in out.get("leagues") or [] if lg in codes]
    out["style_weights"] = {k: float(v) for k, v in (out.get("style_weights") or {}).items()
                            if k in feats and isinstance(v, (int, float))}
    try:
        out["quality_weight"] = float(np.clip(float(out.get("quality_weight") or 0.35), 0, 1))
    except (TypeError, ValueError):
        out["quality_weight"] = 0.35
    return out


def price(row) -> dict:
    """Expected fee range, free-agent status and salary for one pool player."""
    exp = pd.to_datetime(row.contract_expiration_date, errors="coerce")
    years = None if pd.isna(exp) else (exp - pd.Timestamp.now()).days / 365.25
    if years is not None and years <= 0:
        status = "متاح مجانًا (عقده منتهي)"
    elif years is not None and years <= 0.75:
        status = f"ينتهي عقده {exp:%Y-%m} - فرصة مجانية أو بسعر مخفّض"
    else:
        status = f"تحت عقد حتى {exp:%Y-%m}" if years is not None else "مدة العقد غير معروفة"
    base = None
    if pd.notna(row.player_id):
        try:
            fv = valuation.fair_value(int(row.player_id))
            base, sal_lo, sal_hi = fv["base_value"], fv["salary_low"], fv["salary_high"]
        except (KeyError, ValueError):
            base = None
    if base is None:
        base = float(row.market_value_in_eur or 0)
        lo_r, hi_r = WAGE_RATIO.get(row.league, DEFAULT_WAGE_RATIO)
        sal_lo, sal_hi = max(base * lo_r, 50_000), max(base * hi_r, 80_000)
    lo, hi, _ = contract_factor(years)
    fee = (0.0, 0.0) if years is not None and years <= 0 else (base * lo, base * hi)
    return {"status": status, "fee_low": fee[0], "fee_high": fee[1], "salary_low": sal_lo, "salary_high": sal_hi,
            "free": years is not None and years <= 0.75}


def run(spec: dict, n: int = 20) -> pd.DataFrame:
    res = pool.search(spec, n=n)
    if res.empty:
        return res
    prices = [price(r) for r in res.itertuples()]
    for k in ("status", "fee_low", "fee_high", "salary_low", "salary_high", "free"):
        res[k] = [p[k] for p in prices]
    return res


def key_stats(row, spec: dict, k: int = 3) -> str:
    feats = sorted(spec.get("style_weights", {}).items(), key=lambda kv: -abs(kv[1]))[:k] or \
        [("xa90", 1), ("dribbles90", 1), ("recoveries90", 1)]
    out = []
    for f, _ in feats:
        v = getattr(row, f, None)
        if v is not None and not pd.isna(v):
            out.append(f"{pool.FEATURE_AR[f]}: {v:.2f}" if f != "topspeed" else f"{pool.FEATURE_AR[f]}: {v:.1f} كم/س")
    return " · ".join(out)


EXPLAIN_SYSTEM = """You are a neutral scout. For each candidate, write ONE objective Arabic sentence (max 25 words) \
explaining how he matches the recruiter's request, citing his numbers. No hype. Return JSON {"reasons": {"<id>": "..."}}."""


def explain(request: str, res: pd.DataFrame, spec: dict) -> dict[str, str]:
    feats = list(spec.get("style_weights", {})) or ["xa90", "dribbles90", "recoveries90"]
    rows = [{"id": str(r.fotmob_id), "name": r.name, "team": r.team, "league": LEAGUE_NAMES_AR.get(r.league, r.league),
             "position": POSITION_NAMES_AR.get(r.pos_group), "age": None if pd.isna(r.age) else round(r.age),
             "rating": r.rating, "match_pct": r.match,
             "per90": {pool.FEATURE_AR[f]: None if pd.isna(getattr(r, f)) else round(float(getattr(r, f)), 2) for f in feats}}
            for r in res.itertuples()]
    out, _ = ai.chat(EXPLAIN_SYSTEM, f"REQUEST: {request}\n\nCANDIDATES:\n{json.dumps(rows, ensure_ascii=False)}",
                     max_tokens=3000, temperature=0.2)
    return {str(k): str(v) for k, v in (out.get("reasons") or {}).items()}
