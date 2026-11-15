"""Web research: multi-angle search (analysts, fans, technical reports, matches, stats, coach/club) + page text."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/128.0 Safari/537.36", "Accept-Language": "ar,en;q=0.8"}
BACKENDS = "bing, yahoo, yandex"  # DuckDuckGo itself is unreachable from this network
SKIP_FETCH = ("tiktok.com", "youtube.com", "youtu.be", "facebook.com", "instagram.com", "pinterest.", "snapchat.com")
# Video-game rating sites are used as one internal reference among many; they are never shown to the user.
INTERNAL_ONLY = ("ea.com", "futbin.", "sofifa.", "fifaindex.", "futwiz.", "fifacm.", "fifplay.", "fut.gg", "easports")

CATEGORIES = {
    "analysts": "آراء المحللين",
    "fans": "آراء الجمهور",
    "technical": "تقارير فنية",
    "matches": "المباريات",
    "stats": "الإحصاءات والتقييمات",
    "career": "المسيرة والإصابات",
    "contract": "العقد والراتب",
    "marketing": "الانتشار والتسويق",
    "club": "النادي والمدرب",
}


def is_internal(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(d in host for d in INTERNAL_ONLY)


LANGUAGES = {"fr": "الفرنسية", "es": "الإسبانية", "it": "الإيطالية", "de": "الألمانية", "pt": "البرتغالية",
             "tr": "التركية", "nl": "الهولندية"}
LANG_QUERIES = {
    "fr": [("analysts", "{name} analyse points forts points faibles"), ("technical", "{name} profil style de jeu intensité")],
    "es": [("analysts", "{name} análisis puntos fuertes y débiles"), ("technical", "{name} estilo de juego perfil")],
    "it": [("analysts", "{name} analisi punti di forza e debolezze"), ("technical", "{name} caratteristiche stile di gioco")],
    "de": [("analysts", "{name} Stärken Schwächen Analyse"), ("technical", "{name} Spielweise Profil")],
    "pt": [("analysts", "{name} análise pontos fortes e fracos"), ("technical", "{name} estilo de jogo características")],
    "tr": [("analysts", "{name} güçlü ve zayıf yönleri analiz"), ("technical", "{name} oyun tarzı özellikleri")],
    "nl": [("analysts", "{name} sterke en zwakke punten analyse"), ("technical", "{name} speelstijl profiel")],
}
LEAGUE_LANG = {"FR1": "fr", "BE1": "fr", "ES1": "es", "MEX1": "es", "ARG1": "es", "IT1": "it", "L1": "de", "A1": "de",
               "C1": "de", "PO1": "pt", "BRA1": "pt", "TR1": "tr", "NL1": "nl"}
NATIONALITY_LANG = {
    "fr": ["France", "Algeria", "Morocco", "Tunisia", "Senegal", "Cote d'Ivoire", "Ivory Coast", "Cameroon", "Mali",
           "Guinea", "Belgium", "DR Congo", "Burkina Faso", "Gabon", "Benin", "Haiti"],
    "es": ["Spain", "Argentina", "Uruguay", "Colombia", "Mexico", "Chile", "Peru", "Venezuela", "Ecuador", "Paraguay",
           "Bolivia", "Costa Rica", "Honduras", "Panama"],
    "pt": ["Portugal", "Brazil", "Angola", "Cape Verde", "Mozambique", "Guinea-Bissau"],
    "it": ["Italy"], "de": ["Germany", "Austria", "Switzerland"], "tr": ["Turkey", "Türkiye"], "nl": ["Netherlands", "Suriname"],
}


def auto_languages(nationality: str | None, leagues: list[str], limit: int = 3) -> list[str]:
    """Languages the player is most written about in: his nationality first, then the leagues he played in."""
    out = [lang for lang, countries in NATIONALITY_LANG.items() if nationality and any(c in nationality for c in countries)]
    for lg in reversed(leagues):  # most recent leagues first
        lang = LEAGUE_LANG.get(lg)
        if lang and lang not in out:
            out.append(lang)
    return out[:limit]


def build_queries(name: str, club: str | None, flags: list[str], target_club: str | None,
                  coach: str | None, langs: list[str] | None = None) -> list[tuple[str, str]]:
    q = [
        ("technical", f"{name} scouting report strengths weaknesses playing style"),
        ("technical", f"{name} تحليل فني نقاط القوة والضعف"),
        ("analysts", f"{name} analysis form tactical role"),
        ("analysts", f"{name} رأي المحللين مستوى"),
        ("fans", f"{name} fans opinion"),
        ("fans", f"{name} الجماهير"),
        ("matches", f"{name} performance match rating 2026"),
        ("matches", f"{name} أداء مباراة"),
        ("stats", f"{name} pace speed defending ratings attributes"),
        ("stats", f"{name} stats season 2025/26"),
        ("career", f"{name} injury history"),
        ("contract", f"{name} salary contract wages"),
        ("contract", f"راتب {name}"),
        ("marketing", f"{name} instagram followers sponsorship"),
        # Concrete playing traits the summary reports on.
        ("technical", f"{name} work rate pressing defensive contribution"),
        ("technical", f"{name} intensity running stamina criticism physical"),
        ("technical", f"{name} under pressure tight spaces dribbling first touch"),
        ("technical", f"{name} ball carrying progressive runs keeps possession"),
        ("technical", f"{name} speed counter attack space behind"),
        ("analysts", f"{name} المجهود البدني السرعة نقاط الضعف"),
    ]
    for lang in langs or []:
        q += [(cat, tpl.replace("{name}", name)) for cat, tpl in LANG_QUERIES.get(lang, [])]
    for flag in flags[:3]:
        season = flag.split(" ")[0]
        q.append(("career", f"{name} {season} {club or ''} form decline reason coach injury"))
    if target_club:
        q += [
            ("club", f"{target_club} head coach 2026 tactics formation"),
            ("club", f"{target_club} مدرب أسلوب اللعب"),
            ("club", f"{target_club} club identity project objectives 2026"),
            ("club", f"{target_club} أهداف النادي الموسم"),
        ]
    if coach:
        q += [("club", f"{coach} coach tactical philosophy pressing build-up formation"),
              ("club", f"{coach} أسلوب المدرب التكتيكي"),
              ("club", f"{coach} wingers full-backs role demands"),
              ("club", f"{coach} current club 2026"),
              # The player's own club first: a coach who is already there is the most likely current match.
              ("club", f"{coach} {target_club or club or ''} coach 2026")]
    return q


def _search(category: str, query: str, n: int = 5) -> list[dict]:
    try:
        rows = DDGS().text(query, max_results=n, backend=BACKENDS)
    except Exception:
        return []
    return [{"category": category, "query": query, "title": r.get("title", ""), "url": r.get("href", ""),
             "snippet": r.get("body", "")} for r in rows if r.get("href")]


def _page_text(url: str, keywords: list[str], limit: int = 2500) -> str:
    if any(d in url for d in SKIP_FETCH):
        return ""
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            return ""
        soup = BeautifulSoup(r.text, "lxml")
    except Exception:
        return ""
    for t in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        t.decompose()
    paras = [re.sub(r"\s+", " ", p.get_text(" ")).strip() for p in soup.find_all(["p", "li", "h2", "h3", "td"])]
    paras = [p for p in paras if len(p) > 40]
    # Prefer paragraphs that actually mention the player / club.
    kws = [k.lower() for k in keywords if k]
    hits = [p for p in paras if any(k in p.lower() for k in kws)] or paras
    out, total = [], 0
    for p in hits:
        if total + len(p) > limit:
            break
        out.append(p)
        total += len(p)
    return "\n".join(out)


def gather(name: str, club: str | None, flags: list[str], target_club: str | None, coach: str | None,
           progress=None, max_pages: int = 28, langs: list[str] | None = None) -> list[dict]:
    queries = build_queries(name, club, flags, target_club, coach, langs)
    max_pages += 2 * len(langs or [])
    with ThreadPoolExecutor(6) as ex:
        results = [r for chunk in ex.map(lambda q: _search(*q), queries) for r in chunk]
    if progress:
        extra = f" بـ {2 + len(langs or [])} لغات" if langs else ""
        progress(f"🔎 {len(results)} نتيجة بحث من {len(queries)} زاوية{extra} (محللون، جمهور، تقارير، مباريات، إحصاءات)")
    seen, picked = set(), []
    per_cat: dict[str, int] = {}
    for r in results:
        if any(d in r["url"] for d in SKIP_FETCH):
            continue
        cap = 6 if r["category"] in ("club", "technical", "analysts") else 3
        if r["url"] in seen or per_cat.get(r["category"], 0) >= cap:
            continue
        seen.add(r["url"])
        per_cat[r["category"]] = per_cat.get(r["category"], 0) + 1
        picked.append(r)
        if len(picked) >= max_pages:
            break
    surname = name.split()[-1]
    keywords = [surname, name, target_club or "", coach or ""]
    with ThreadPoolExecutor(8) as ex:
        texts = list(ex.map(lambda r: _page_text(r["url"], keywords), picked))
    for r, t in zip(picked, texts):
        r["text"] = t
    if progress:
        progress(f"📄 قراءة {sum(1 for t in texts if t)} صفحة")
    # Keep snippets of the non-fetched results too - they are cheap evidence.
    extra = [r | {"text": ""} for r in results if r["url"] not in seen][:25]
    return picked + extra


def evidence_text(items: list[dict], max_chars: int = 60000) -> str:
    blocks, total = [], 0
    for i, r in enumerate(items, 1):
        body = r.get("text") or r.get("snippet") or ""
        block = f"[S{i}] ({CATEGORIES.get(r['category'], r['category'])}) {r['title']}\n{r['url']}\n{body}\n"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks)
