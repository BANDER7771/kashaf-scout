"""Overall rating per position, best positions, and fit with tactical systems."""
from __future__ import annotations

from .config import OUTFIELD_GROUPS, POSITION_WEIGHTS

# Each system: style emphasis (blended into position weights) and the role each position
# group plays in it. A position group missing from "roles" is mapped via "maps_to".
SYSTEMS = [
    {
        "id": "possession", "name": "4-3-3 استحواذ وبناء من الخلف",
        "style": "استحواذ عالٍ، تمرير قصير، ضغط عكسي فوري بعد فقدان الكرة",
        "examples": "مانشستر سيتي (غوارديولا)، أرسنال (أرتيتا)، برشلونة",
        "emphasis": {"passing": .22, "composure": .18, "vision": .14, "tactical_iq": .20, "work_rate": .10, "dribbling": .16},
        "roles": {"ST": "مهاجم مركّب / وهمي", "W": "جناح مقلوب", "AM": "لاعب وسط حر (رقم 8)", "CM": "لاعب وسط تمريري",
                  "DM": "محور بناء (رقم 6)", "FB": "ظهير مقلوب يدخل للعمق", "CB": "قلب دفاع بنّاء", "GK": "حارس يجيد اللعب بالقدم"},
    },
    {
        "id": "gegenpress", "name": "4-2-3-1 / 4-3-3 ضغط عالٍ وتحولات سريعة",
        "style": "ضغط شرس في نصف ملعب الخصم، استعادة سريعة، هجوم عمودي سريع",
        "examples": "ليفربول (كلوب)، لايبزيغ، ليفركوزن",
        "emphasis": {"work_rate": .25, "stamina": .20, "pace": .14, "acceleration": .14, "tactical_iq": .12, "tackling": .15},
        "roles": {"ST": "مهاجم ضاغط", "W": "جناح ضاغط عمودي", "AM": "صانع ألعاب ضاغط", "CM": "لاعب وسط متنقل (box-to-box)",
                  "DM": "محور قاطع للكرات", "FB": "ظهير هجومي عالي", "CB": "قلب دفاع سريع بخط عالٍ", "GK": "حارس متقدم (sweeper)"},
    },
    {
        "id": "counter", "name": "4-4-2 / 4-2-3-1 كتلة متوسطة ومرتدات",
        "style": "تنظيم دفاعي متماسك، ترك الاستحواذ للخصم، مرتدات سريعة ومباشرة",
        "examples": "أتلتيكو مدريد (سيميوني)، ريال مدريد في المباريات الكبيرة",
        "emphasis": {"defensive_awareness": .18, "work_rate": .16, "strength": .10, "pace": .16, "acceleration": .10,
                     "finishing": .15, "tackling": .15},
        "roles": {"ST": "مهاجم مرتدات سريع", "W": "جناح مرتدات بواجبات دفاعية", "AM": "مهاجم ثانٍ", "CM": "لاعب وسط مجتهد",
                  "DM": "محور دفاعي", "FB": "ظهير دفاعي متوازن", "CB": "قلب دفاع صلب", "GK": "حارس تقليدي"},
    },
    {
        "id": "back3", "name": "3-4-2-1 / 3-5-2 ثلاثي دفاعي وأظهرة جناحية",
        "style": "ثلاثة في الدفاع، أظهرة تغطي الرواق كاملاً، ارتباط بين الخطوط",
        "examples": "إنتر (إنزاغي)، تشيلسي (كونتي/توخيل)",
        "emphasis": {"stamina": .20, "pace": .14, "tactical_iq": .20, "passing": .14, "aerial": .10, "strength": .10, "composure": .12},
        "roles": {"ST": "مهاجم محوري", "AM": "لاعب بين الخطوط (رقم 10)", "CM": "لاعب وسط", "DM": "محور ارتكاز",
                  "FB": "ظهير جناح (wing-back)", "CB": "قلب دفاع في ثلاثي", "GK": "حارس"},
        "maps_to": {"W": ["FB", "AM"]},
    },
    {
        "id": "direct", "name": "4-4-2 لعب مباشر وكرات عرضية",
        "style": "كرات طويلة، صراعات هوائية، عرضيات من الأطراف، مهاجم صندوق",
        "examples": "فرق الصراع على البقاء، بيرنلي سابقًا",
        "emphasis": {"aerial": .25, "strength": .25, "pace": .15, "finishing": .15, "work_rate": .10, "stamina": .10},
        "roles": {"ST": "مهاجم صندوق / رأس حربة", "W": "لاعب طرف عرضيات", "AM": "مهاجم ثانٍ", "CM": "لاعب وسط افتكاك",
                  "DM": "لاعب وسط افتكاك", "FB": "ظهير دفاعي", "CB": "قلب دفاع هوائي", "GK": "حارس"},
    },
    {
        "id": "lowblock", "name": "5-4-1 دفاع منخفض",
        "style": "تكتل دفاعي قريب من المرمى، إغلاق المساحات، اعتماد على الكرات الثابتة",
        "examples": "الفرق الأقل إمكانيات أمام الكبار",
        "emphasis": {"defensive_awareness": .25, "tackling": .20, "strength": .15, "aerial": .15, "work_rate": .15, "stamina": .10},
        "roles": {"ST": "مهاجم وحيد يحتفظ بالكرة", "W": "لاعب طرف دفاعي", "CM": "لاعب وسط دفاعي", "DM": "محور دفاعي",
                  "FB": "ظهير جناح دفاعي", "CB": "قلب دفاع في خماسي", "GK": "حارس"},
        "maps_to": {"AM": ["CM", "W"]},
    },
]

POSITION_DISTANCE_PENALTY = {0: 0, 1: 2, 2: 6}
_ADJACENT = {
    "ST": {"W", "AM"}, "W": {"ST", "AM", "FB"}, "AM": {"ST", "W", "CM"}, "CM": {"AM", "DM"},
    "DM": {"CM", "CB"}, "FB": {"W", "CB"}, "CB": {"DM", "FB"}, "GK": set(),
}


def weighted(attrs: dict[str, dict], weights: dict[str, float]) -> float | None:
    """Weighted mean over the attributes we actually have (weights re-normalised)."""
    have = {k: w for k, w in weights.items() if attrs.get(k, {}).get("value") is not None}
    total = sum(have.values())
    if total < 0.5:  # not enough coverage to say anything meaningful
        return None
    return sum(attrs[k]["value"] * w for k, w in have.items()) / total


def familiarity_penalty(natural: str, alternatives: list[str], target: str) -> int:
    if target == natural:
        return 0
    if target in alternatives:
        return POSITION_DISTANCE_PENALTY[1]
    return POSITION_DISTANCE_PENALTY[1] if target in _ADJACENT.get(natural, set()) else POSITION_DISTANCE_PENALTY[2]


def overall_for(attrs: dict, group: str) -> float | None:
    """Position overall: weighted mean, pulled up towards the player's best key skills (like a scout would rate
    a specialist) rather than flattened by secondary attributes."""
    weights = POSITION_WEIGHTS[group]
    base = weighted(attrs, weights)
    if base is None:
        return None
    key = sorted((attrs[k]["value"] for k in weights if attrs.get(k, {}).get("value") is not None), reverse=True)[:3]
    return 0.65 * base + 0.35 * (sum(key) / len(key))


def position_overalls(attrs: dict, natural: str, alternatives: list[str]) -> list[dict]:
    """Natural position first, then only positions he has actually played (no invented roles)."""
    out = []
    for g in [natural] + [a for a in alternatives if a != natural]:
        score = overall_for(attrs, g)
        if score is None:
            continue
        pen = familiarity_penalty(natural, alternatives, g)
        out.append({"group": g, "raw": round(score), "score": round(score - pen), "penalty": pen,
                    "natural": g == natural, "alternative": g in alternatives})
    return out[:1] + sorted(out[1:], key=lambda r: r["score"], reverse=True)


def system_fit(attrs: dict, natural: str, alternatives: list[str]) -> list[dict]:
    results = []
    for sys in SYSTEMS:
        # Judge him in his own position whenever the system has it; fall back to the mapped role otherwise
        # (e.g. a winger in a back-three system plays wing-back or No.10).
        if natural in sys["roles"]:
            candidates = [natural]
        else:
            mapped = sys.get("maps_to", {}).get(natural, [])
            candidates = [a for a in mapped if a in alternatives] or mapped[:1]
        best = None
        for g in candidates:
            base = POSITION_WEIGHTS[g]
            emph = {k: v for k, v in sys["emphasis"].items() if k in base or g != "GK"}
            weights = {k: 0.6 * base.get(k, 0) + 0.4 * emph.get(k, 0) for k in set(base) | set(emph)}
            score = weighted(attrs, weights)
            if score is None:
                continue
            key = sorted((attrs[k]["value"] for k in weights if attrs.get(k, {}).get("value") is not None),
                         reverse=True)[:3]
            score = 0.65 * score + 0.35 * (sum(key) / len(key))
            score -= familiarity_penalty(natural, alternatives, g)
            if best is None or score > best["score"]:
                best = {"group": g, "score": score}
        if best is None:
            continue
        fit = max(0.0, min(1.0, (best["score"] - 45) / (88 - 45))) * 100
        results.append({"id": sys["id"], "name": sys["name"], "style": sys["style"], "examples": sys["examples"],
                        "role": sys["roles"][best["group"]], "group": best["group"],
                        "score": round(best["score"]), "fit": round(fit)})
    return sorted(results, key=lambda r: r["fit"], reverse=True)
