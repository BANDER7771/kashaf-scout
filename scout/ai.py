"""DeepSeek analysis: merges computed data with multi-source web evidence into an objective Arabic scouting report."""
from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from . import settings
from .config import CACHE_DIR, DEEPSEEK_BASE_URL
from .metrics import TRAITS

POSITIONS = ["ST", "W", "AM", "CM", "DM", "FB", "CB", "GK"]
PHASES = ["بداية", "صعود", "ذروة", "ثبات", "تراجع", "انتعاش", "إصابة", "انتقال"]
COACH_VERDICTS = ["يناسب", "يناسب جزئيًا", "لا يناسب"]
LEVELS = ["قوي", "مقبول", "ضعيف"]


class AIError(RuntimeError):
    pass


def available() -> bool:
    return settings.deepseek_key() is not None


def chat(system: str, user: str, max_tokens: int = 8000, temperature: float = 0.2) -> tuple[dict, dict]:
    key = settings.deepseek_key()
    if not key:
        raise AIError("مفتاح DeepSeek غير مضبوط.")
    body = {"model": settings.deepseek_model(), "max_tokens": max_tokens, "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for attempt in range(3):
        try:
            r = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions", json=body, timeout=300,
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        except requests.RequestException as e:
            if attempt == 2:
                raise AIError("تعذّر الاتصال بـ DeepSeek - تحقق من الإنترنت أو من خروج الشبكة على الخادم.") from e
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 401:
            raise AIError("مفتاح DeepSeek غير صالح - تحقق من قيمة DEEPSEEK_API_KEY على الخادم.")
        if r.status_code == 402:
            raise AIError("رصيد حساب DeepSeek غير كافٍ - اشحن الحساب من platform.deepseek.com ثم أعد المحاولة.")
        if r.status_code in (429, 500, 502, 503) and attempt < 2:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise AIError(f"خطأ من DeepSeek ({r.status_code}): {r.text[:200]}")
        data = r.json()
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise AIError("انقطع رد DeepSeek قبل اكتماله - جرّب مرة أخرى.")
        return _parse_json(choice["message"]["content"]), data.get("usage", {})
    raise AIError("DeepSeek مشغول حاليًا - حاول بعد قليل.")


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise AIError("تعذّر قراءة رد DeepSeek.")


# ---------------------------------------------------------------- prompts

OBJECTIVITY = """Objectivity rules (most important):
- You are a neutral analyst, not a promoter and not a critic. Neither optimistic nor pessimistic: describe what the \
data and the evidence actually support, with calibrated confidence.
- Order of trust: measured match data in PLAYER DATA > match reports and technical/analyst reports > statistics \
sites > fan opinion. Treat fan opinion as sentiment, never as fact. Recent seasons outweigh old ones.
- Every strength must be backed by data or at least one credible source; state its limits when relevant. Every \
weakness must be backed the same way; give its context (team style, role, league, age).
- Adjust for context: league strength (league_strength_vs_premier_league_1_0), team dominance \
(team_context_last_league_season - a top team inflates attacking output and deflates defensive volume), role and \
minutes, and age (age_context: players past their peak decline physically first; rate CURRENT level, not peak).
- Each line in match_stats_by_season gives the VALUE first (e.g. "دقة التمرير: القيمة 84.3%") and then the RANK \
("ترتيبه: أفضل من 69% من لاعبي مركزه في الدوري"). A rank is NOT an accuracy or success rate: write values as values \
and ranks as "أفضل من X% من لاعبي مركزه". Example: pass accuracy 84.3% with rank 69 -> "دقة تمريره 84.3%، أفضل من \
69% من لاعبي مركزه" - never "دقة تمريره 69%". The same applies to playing_traits_from_data_0_100.
- No superlatives (أفضل، أسطوري، استثنائي، عالمي) unless the data shows a top-5% percentile or major honours confirm it.
- If evidence is thin or contradictory, say so and lower confidence. Never fill gaps with assumptions.
- Never invent numbers; every number must appear in PLAYER DATA or EVIDENCE."""

STYLE = """Style rules:
- Clear Modern Standard Arabic in every text value; names in Arabic script. Avoid English words (xG/xA allowed).
- Never mention video games or game-rating sites (FIFA, EA FC, FUT, SoFIFA, FUTBIN...). If such ratings informed \
you, use them silently as one reference among others.
- Never write source tags like [S3] in the text.
- The player's natural position is "{position}" (from actual line-ups). Describe him as that; mention other \
positions only as secondary options backed by positions_played_with_games.
- Concise: each paragraph at most 80 words, lists 3-6 items, each list item at most 25 words.
- Output one JSON object exactly matching the template."""

TRAIT_GUIDE = """Playing traits: playing_traits_from_data_0_100 is a MEASURED BASELINE, not the answer. Score every \
trait from ALL the evidence. Free data has no running-distance, sprint or speed tracking, so for physical_effort \
(running volume, intensity, recovery runs, pressing without the ball) and open_space (speed in transition) rely mostly \
on match and scouting reports - if reports describe low intensity or limited running, the score must be low even if \
the baseline is average. Ball carrying and ball retention must reflect how often he progresses with the ball and \
keeps it under pressure (dribble success, touches, pass accuracy, reports), not how many times he loses it in total.
Trait scores must agree with pros and cons: a trait you list as a con scores 40 or less, a pro 65 or more. Never copy the baseline numbers blindly. A trait cannot be both a pro and a con.
Seasons marked "small sample" (under 900 minutes) are early-season snapshots: mention them only as recent form and base your judgements on the most recent full season."""

PROFILE_TEMPLATE = {
    "overall": "int 0-99, current level on a global scale (88-94 world class, 80-87 elite starter in a top league, "
               "72-79 solid starter, 64-71 squad player, below 64 limited)",
    "potential": "int 0-99, realistic ceiling given age",
    "playing_style": "Arabic paragraph: how he plays, where he receives, what he does with and without the ball",
    "position_and_system": "Arabic paragraph: the role and game model that get the best out of him, and those that do not",
    "technical_value": "Arabic paragraph: what he would add technically",
    "marketing": {"score": "int 0-100", "social_reach": "Arabic: followers/markets with numbers from evidence",
                  "analysis": "Arabic paragraph: commercial value, markets, sponsors"},
    "reported_salary": "Arabic: reported salary with the source name, or 'غير متوفر'",
}

SUMMARY_TEMPLATE = {
    "summary": {
        "headline": "Arabic, one objective sentence (max 25 words) describing who he is now",
        "archetype": "Arabic, 3-6 words profile label e.g. 'لاعب وسط يتقدم بالكرة ويصنع'",
        "confidence": "عالية|متوسطة|منخفضة - how well the evidence supports this analysis",
        "data_gaps": ["Arabic: important things the data/evidence could not establish"],
    },
    "pros": [{"title": "Arabic, 2-5 words e.g. 'التقدم بالكرة والاحتفاظ بها'",
              "detail": "Arabic, max 35 words: what he concretely does on the pitch + the number or source behind it",
              "trait": f"closest key from {list(TRAITS)} or ''"}],
    "cons": [{"title": "Arabic, 2-5 words e.g. 'المجهود البدني والركض'",
              "detail": "Arabic, max 35 words: what concretely goes wrong on the pitch + the number or source behind it",
              "trait": f"closest key from {list(TRAITS)} or ''"}],
    "traits": [{"key": f"one of {list(TRAITS)}", "score": "int 0-100 (50 = average for his position)",
                "explanation": "Arabic, max 30 words"}],
    "suitable_teams": [{"profile": "Arabic: type of team/game model that suits him (not a specific club)",
                        "why": "Arabic, one sentence"}],
    "unsuitable_teams": [{"profile": "Arabic: type of team/game model that does not suit him", "why": "Arabic"}],
    "verdict": "Arabic, 2-3 balanced sentences for a sporting director",
    "risks": ["Arabic"],
    "sources_summary": {"analysts": "Arabic", "fans": "Arabic (sentiment only)", "technical": "Arabic",
                        "matches": "Arabic", "stats": "Arabic"},
}

CAREER_TEMPLATE = {
    "career_phases": [{"period": "e.g. 2019-2021", "phase": f"one of {PHASES}",
                       "explanation": "Arabic: what happened and WHY (coach, injury, role, transfer, age, team)"}],
    "consistency_analysis": "Arabic paragraph interpreting the consistency score and its parts",
    "decline_analysis": "Arabic paragraph: is there a real decline (performance vs value vs minutes) and why",
    "valuation_view": "Arabic paragraph on fair fee and salary ranges vs market values and reported salary",
    "coach_fit": "null when no TARGET CLUB and no COACH; otherwise an object (see COACH TEMPLATE)",
}
COACH_TEMPLATE = {
    "coach_name": "Arabic: the coach (given, or the current head coach of the target club found in evidence)",
    "club_name": "Arabic: target club, or the coach's current club",
    "formation": "e.g. 4-3-3",
    "coach_methods": [{"principle": "Arabic: e.g. 'ضغط عالٍ بعد فقدان الكرة'", "detail": "Arabic, one sentence"}],
    "role_for_player": "Arabic: the role this coach would give him",
    "requirements": [{"requirement": "Arabic: what the coach demands from this position",
                      "player_level": f"one of {LEVELS}", "note": "Arabic: evidence from the player's data"}],
    "verdict": f"one of {COACH_VERDICTS}",
    "fit_score": "int 0-100",
    "fit_analysis": "Arabic paragraph comparing the coach's methods with the player's profile, pros and cons",
    "club_identity": "Arabic or empty string if no club",
    "club_objectives": "Arabic or empty string if no club",
}

PROFILE_SYSTEM = ("You are a neutral senior scout describing how a player plays and his overall level.\n\n"
                  + OBJECTIVITY + "\n\n" + STYLE)
SUMMARY_SYSTEM = ("You are a neutral senior scout writing the executive summary of a player report. pros and cons "
                  "must together cover every clear positive and negative across four sides: physical (running volume, "
                  "intensity, speed, duels), with the ball (carrying, retention under pressure, passing range, "
                  "crossing, vision, finishing), without the ball (pressing, positioning, defensive work) and mental "
                  "(consistency, big games). Do not skip a known negative because the data baseline looks average. "
                  "IDENTITY: summary.archetype, summary.headline and the first pros must describe what most "
                  "defines him - start from signature_strengths (his highest-ranked qualities in the data) and "
                  "confirm with evidence. Secondary qualities, and qualities mentioned in SCOUT OBSERVATIONS, add "
                  "nuance further down but must not replace his core identity unless the data agrees. "
                  "Order each list from most to least important (4-6 pros, 3-5 cons). Also say which TYPES of team "
                  "he suits.\n\n" + TRAIT_GUIDE + "\n\n" + OBJECTIVITY + "\n\n" + STYLE)

CAREER_SYSTEM = ("You are a neutral senior scout writing the career, consistency and value sections. Career phases "
                 "must be chronological, cover the whole career, and explain causes from EVIDENCE; if a cause is "
                 "unknown say so. If a TARGET CLUB or a COACH is given, analyse the coach: identify him (current head "
                 "coach of the target club if not given), list his real methods from evidence (if evidence about him "
                 "is thin, say so; coaches move clubs, so when sources disagree about his current club trust the most "
                 "recent dated source, and if evidence shows him coaching the player's own club, that is his current club), "
                 "list what he demands from the player's position, compare each demand with the "
                 "player's measured profile, and give a clear verdict. role_for_player MUST be consistent with the "
                 "formation: if the formation has no slot for the player's natural position (e.g. a winger in 3-5-2), "
                 "say which role he would have to play instead and judge the fit on that role.\n\n"
                 + OBJECTIVITY + "\n\n" + STYLE)


def _user_msg(template: dict, data: dict, evidence: str, target_club: str | None, coach: str | None,
              notes: str | None = None) -> str:
    scout = (f"SCOUT OBSERVATIONS (written by the user, who watches this player regularly - credible eye-test "
             f"evidence about what free data cannot measure, such as running volume and intensity. Integrate them "
             f"with the data, reflect them in pros/cons and trait scores, and say so if the data clearly contradicts "
             f"one of them):\n{notes}\n\n") if notes else ""
    return (f"JSON TEMPLATE:\n{json.dumps(template, ensure_ascii=False, indent=1)}\n\n"
            f"TARGET CLUB: {target_club or 'none'}\nCOACH: {coach or 'none given'}\n\n" + scout
            + f"PLAYER DATA:\n{json.dumps(data, ensure_ascii=False, default=str)}\n\n"
            f"EVIDENCE (web sources):\n{evidence or 'No web evidence available - rely on PLAYER DATA and lower confidence.'}")


def analyse(data: dict, evidence: str, position: str, target_club: str | None, coach: str | None,
            notes: str | None = None) -> dict:
    career_template = dict(CAREER_TEMPLATE)
    if target_club or coach:
        career_template["coach_fit"] = COACH_TEMPLATE
    jobs = [
        (PROFILE_SYSTEM.replace("{position}", position), PROFILE_TEMPLATE),
        (SUMMARY_SYSTEM.replace("{position}", position), SUMMARY_TEMPLATE),
        (CAREER_SYSTEM.replace("{position}", position), career_template),
    ]
    with ThreadPoolExecutor(3) as ex:
        results = list(ex.map(lambda j: chat(j[0], _user_msg(j[1], data, evidence, target_club, coach, notes)), jobs))
    merged, usage = {}, {"prompt_tokens": 0, "completion_tokens": 0}
    for part, u in results:
        merged.update(part)
        for k in usage:
            usage[k] += u.get(k, 0)
    out = _normalise(merged, position, bool(target_club or coach))
    out["_usage"] = usage
    return out


def _int(x, lo=0, hi=99, default=None):
    try:
        return int(max(lo, min(hi, round(float(x)))))
    except (TypeError, ValueError):
        return default


def _strip_refs(text) -> str:
    return re.sub(r"\s*[\[(]\s*S\d+(?:\s*[,،]\s*S\d+)*\s*[\])]", "", str(text or "")).strip()


def _normalise(o: dict, position: str, has_coach: bool) -> dict:
    o.pop("attributes", None)
    o["overall"] = _int(o.get("overall"))
    o["potential"] = _int(o.get("potential"))
    for k in ("pros", "cons"):
        o[k] = [{"title": _strip_refs(x.get("title")), "detail": _strip_refs(x.get("detail")),
                 "trait": x.get("trait") if x.get("trait") in TRAITS else ""}
                for x in o.get(k) or [] if isinstance(x, dict) and x.get("title")][:6]
    for k in ("playing_style", "position_and_system", "technical_value", "reported_salary", "consistency_analysis",
              "decline_analysis", "valuation_view", "verdict"):
        o[k] = _strip_refs(o.get(k))
    for k in ("strengths", "weaknesses", "risks"):
        o[k] = [_strip_refs(x) for x in (o.get(k) or [])][:8]
    sm = o.get("summary") or {}
    o["summary"] = {"headline": _strip_refs(sm.get("headline")), "archetype": _strip_refs(sm.get("archetype")),
                    "confidence": sm.get("confidence") if sm.get("confidence") in ("عالية", "متوسطة", "منخفضة") else "متوسطة",
                    "data_gaps": [_strip_refs(x) for x in sm.get("data_gaps") or []][:5]}
    o["traits"] = [{"key": t.get("key"), "score": _int(t.get("score"), 0, 100), "explanation": _strip_refs(t.get("explanation"))}
                   for t in o.get("traits") or [] if isinstance(t, dict) and t.get("key") in TRAITS]
    for k in ("suitable_teams", "unsuitable_teams"):
        o[k] = [{"profile": _strip_refs(t.get("profile")), "why": _strip_refs(t.get("why"))}
                for t in o.get(k) or [] if isinstance(t, dict) and t.get("profile")][:4]
    m = o.get("marketing") or {}
    o["marketing"] = {"score": _int(m.get("score"), 0, 100, 0), "social_reach": _strip_refs(m.get("social_reach")),
                      "analysis": _strip_refs(m.get("analysis"))}
    o["career_phases"] = [{"period": str(p.get("period", "")), "phase": p.get("phase") if p.get("phase") in PHASES else "ثبات",
                           "explanation": _strip_refs(p.get("explanation"))}
                          for p in (o.get("career_phases") or []) if isinstance(p, dict)]
    o["sources_summary"] = {k: _strip_refs(v) for k, v in (o.get("sources_summary") or {}).items() if v}
    cf = o.get("coach_fit") if has_coach and isinstance(o.get("coach_fit"), dict) else None
    if cf:
        cf["fit_score"] = _int(cf.get("fit_score"), 0, 100)
        cf["verdict"] = cf.get("verdict") if cf.get("verdict") in COACH_VERDICTS else "يناسب جزئيًا"
        cf["coach_methods"] = [{"principle": _strip_refs(m.get("principle")), "detail": _strip_refs(m.get("detail"))}
                               for m in cf.get("coach_methods") or [] if isinstance(m, dict)]
        cf["requirements"] = [{"requirement": _strip_refs(r.get("requirement")),
                               "player_level": r.get("player_level") if r.get("player_level") in LEVELS else "مقبول",
                               "note": _strip_refs(r.get("note"))}
                              for r in cf.get("requirements") or [] if isinstance(r, dict)]
        for k in ("coach_name", "club_name", "formation", "role_for_player", "fit_analysis", "club_identity", "club_objectives"):
            cf[k] = _strip_refs(cf.get(k))
    o["coach_fit"] = cf
    o.pop("club", None)
    return o


# ---------------------------------------------------------------- cache

def _cache_path(player_id: int, target_club: str | None, coach: str | None, notes: str | None = None):
    slug = re.sub(r"[^\w]+", "_", f"{target_club or 'none'}_{coach or ''}".lower()).strip("_")
    if notes and notes.strip():
        slug += "_n" + hashlib.md5(notes.strip().encode("utf-8")).hexdigest()[:8]
    return CACHE_DIR / "ai" / f"v3_{player_id}_{slug}.json"


def load_cached(player_id: int, target_club: str | None, coach: str | None, notes: str | None = None) -> dict | None:
    p = _cache_path(player_id, target_club, coach, notes)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_cached(player_id: int, target_club: str | None, coach: str | None, payload: dict,
                notes: str | None = None) -> None:
    p = _cache_path(player_id, target_club, coach, notes)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "_saved_at": time.strftime("%Y-%m-%d %H:%M")}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def transliterate(arabic_name: str) -> list[str]:
    """Arabic player name -> likely Latin spellings (for searching the Latin-script databases)."""
    out, _ = chat('Return JSON {"names": [..]} with up to 4 Latin-script spellings that Transfermarkt would most '
                  "likely use for the footballer the user names. Most likely first.", arabic_name, max_tokens=300,
                  temperature=0.1)
    return [n for n in out.get("names", []) if isinstance(n, str)][:4]
