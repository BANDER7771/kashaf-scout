"""Design system for كشّاف (light, editorial): CSS tokens and HTML/SVG components."""
from __future__ import annotations

import html
import math
import re

import numpy as np

from scout.config import LEAGUE_NAMES_AR, POSITION_NAMES_AR

CSS = """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Tajawal:wght@500;700;800;900&display=swap" rel="stylesheet">
<style>
:root { --bg:#F5F7FB; --card:#FFFFFF; --ink:#0F172A; --ink2:#334155; --muted:#64748B; --line:#E5E9F0; --soft:#F1F4F9;
  --navy:#0B1222; --navy2:#13203A; --accent:#16A34A; --accent2:#0EA5E9; --gold:#C99A2E; --red:#DC2626;
  --shadow:0 1px 2px rgba(15,23,42,.04), 0 10px 28px rgba(15,23,42,.06); }
html, body, [class*="css"], p, li, label, input, textarea, button, div, span { font-family:'IBM Plex Sans Arabic',sans-serif; }
h1,h2,h3,h4 { font-family:'Tajawal',sans-serif !important; color:var(--ink); }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.main .block-container, [data-testid="stMainBlockContainer"] { direction:rtl; text-align:right; padding-top:1.1rem; max-width:1240px; }
[data-testid="stSidebar"] { direction:rtl; text-align:right; }
p { line-height:1.95; color:var(--ink2); }
/* streamlit containers -> cards */
/* only st.container(border=True): plain layout wrappers carry the empty emotion class st-emotion-cache-0 */
[data-testid="stVerticalBlockBorderWrapper"]:not(.st-emotion-cache-0) { background:var(--card);
  border:1px solid var(--line) !important; border-radius:18px !important; box-shadow:var(--shadow); }
[data-testid="stExpander"] details { background:var(--card); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow); }
[data-testid="stExpander"] summary p { font-weight:700; color:var(--ink); }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { background:#fff; border:1px solid var(--line); border-radius:12px; }
.stButton button, .stDownloadButton button { border-radius:12px; font-weight:700; border:1px solid var(--line); }
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] { background:var(--navy); color:#fff; border:0; }
.stButton button[kind="primary"]:hover, .stDownloadButton button[kind="primary"]:hover { background:var(--navy2); color:#fff; }
/* radios -> underline tabs */
[data-testid="stRadio"] [role="radiogroup"] { gap:4px; flex-wrap:wrap; border-bottom:1px solid var(--line); }
[data-testid="stRadio"] [role="radiogroup"] label { padding:9px 16px !important; margin:0 !important; border-bottom:3px solid transparent; border-radius:0; }
[data-testid="stRadio"] [role="radiogroup"] label p { color:var(--muted); font-weight:600; font-size:15px; }
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) { border-bottom-color:var(--accent); }
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p { color:var(--ink); font-weight:800; }
[data-testid="stRadio"] [role="radiogroup"] label > div:first-child { display:none; }
/* top bar */
.ks-top { display:flex; align-items:center; justify-content:space-between; padding:6px 4px 10px; }
.ks-logo { display:flex; align-items:center; gap:12px; }
.ks-logo .mark { width:46px; height:46px; } .ks-logo .mark svg { width:46px; height:46px; display:block; filter:drop-shadow(0 6px 14px rgba(11,18,34,.25)); }
.ks-logo .name { font-family:'Tajawal'; font-size:28px; font-weight:900; color:var(--ink); line-height:1; }
.ks-logo .tag { color:var(--muted); font-size:13px; margin-top:4px; }
.ks-chip { display:inline-flex; align-items:center; gap:6px; padding:4px 12px; border-radius:999px; font-size:12.5px;
  background:var(--card); border:1px solid var(--line); color:var(--ink2); margin:2px; }
.ks-chip.ok { color:#15803D; border-color:#BBF7D0; background:#F0FDF4; } .ks-chip.off { color:#B91C1C; border-color:#FECACA; background:#FEF2F2; }
/* hero band */
.ks-hero { background:linear-gradient(120deg,#0B1222 0%,#13203A 100%);
  color:#E6EAF2; border-radius:22px; padding:26px 28px; box-shadow:0 18px 40px rgba(11,18,34,.22); }
.ks-hero .nm { font-family:'Tajawal'; font-weight:900; font-size:36px; line-height:1.1; color:#fff; }
.ks-hero .sub { color:#AEB8CB; margin:6px 0 12px; font-size:14px; }
.ks-hero .arch { color:#86EFAC; font-weight:800; font-size:19px; }
.ks-hero .head { color:#D7DEEA; font-size:15.5px; line-height:1.9; margin:6px 0 12px; }
.ks-hero .gl { display:inline-block; padding:4px 12px; margin:2px; border-radius:999px; font-size:12.5px; background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.14); color:#E6EAF2; }
.ks-kpis { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-top:16px; }
.ks-kpi { background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12); border-radius:14px; padding:10px 12px; }
.ks-kpi .l { color:#AEB8CB; font-size:11.5px; } .ks-kpi .v { font-family:'Tajawal'; font-weight:800; font-size:18px; color:#fff; margin-top:2px; }
.ks-kpi .s { color:#8C97AD; font-size:11px; }
/* money ranges inside Arabic text: let each value pick its own direction */
.ks-kpi .v, .ks-tile .v, .ks-price b, .kc-foot .v { unicode-bidi:plaintext; }
.ks-verdict { background:#F0FDF4; border:1px solid #BBF7D0; border-radius:16px; padding:14px 18px; margin:14px 0 4px; line-height:1.95; color:var(--ink2); }
/* sections & cards */
.ks-sec { display:flex; align-items:center; gap:10px; margin:28px 0 12px; }
.ks-sec h3 { margin:0; font-family:'Tajawal'; font-weight:800; font-size:21px; color:var(--ink); }
.ks-sec .dot { width:6px; height:22px; border-radius:4px; background:var(--accent); }
.ks-sec small { color:var(--muted); font-size:13px; }
.ks-panel { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:18px 20px; margin-bottom:12px; box-shadow:var(--shadow); }
.ks-muted { color:var(--muted); font-size:13px; }
.ks-list { margin:0; padding-inline-start:18px; } .ks-list li { margin:5px 0; line-height:1.85; color:var(--ink2); }
.ks-pc { background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:13px 15px; margin-bottom:10px; box-shadow:var(--shadow); }
.ks-pc .top { display:flex; justify-content:space-between; align-items:center; gap:8px; }
.ks-pc .tt { font-weight:800; font-size:15.5px; color:var(--ink); } .ks-pc .dd { color:var(--ink2); font-size:13.5px; margin-top:5px; line-height:1.85; }
.ks-score { font-family:'Tajawal'; font-weight:900; font-size:14px; min-width:38px; text-align:center; border-radius:8px; color:#fff; padding:1px 7px; }
.ks-team { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:12px 14px; margin-bottom:10px; box-shadow:var(--shadow); }
.ks-team b { display:block; margin-bottom:3px; color:var(--ink); } .ks-team span { color:var(--muted); font-size:13px; line-height:1.8; }
.ks-badge { display:inline-block; padding:3px 12px; border-radius:999px; font-weight:800; font-size:13px; }
.ks-tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin-bottom:12px; }
.ks-tile { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:12px 14px; box-shadow:var(--shadow); }
.ks-tile .l { color:var(--muted); font-size:12.5px; } .ks-tile .v { font-family:'Tajawal'; font-size:20px; font-weight:800; color:var(--ink); }
.ks-tile .s { font-size:12px; color:var(--muted); }
/* bars */
.ks-attr { display:grid; grid-template-columns:170px 1fr 48px; gap:12px; align-items:center; margin:8px 0; }
.ks-attr .lb { font-size:14px; color:var(--ink2); }
.ks-bar { height:8px; background:var(--soft); border-radius:6px; overflow:hidden; } .ks-bar > i { display:block; height:100%; border-radius:6px; }
.ks-val { font-family:'Tajawal'; font-weight:800; font-size:15px; text-align:center; border-radius:8px; color:#fff; }
/* rings */
.ks-ring { width:104px; height:104px; border-radius:50%; margin:auto; display:grid; place-items:center; background:var(--soft); }
.ks-ring > div { width:82px; height:82px; border-radius:50%; background:#fff; display:grid; place-items:center; }
.ks-ring .v { font-family:'Tajawal'; font-size:30px; font-weight:900; line-height:1; text-align:center; color:var(--ink); }
.ks-ring .l { font-size:11px; color:var(--muted); text-align:center; }
/* traits grid */
.ks-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:12px; }
.ks-trait { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:12px 14px; box-shadow:var(--shadow); }
.ks-trait .top { display:flex; justify-content:space-between; align-items:center; } .ks-trait .nm { font-weight:700; color:var(--ink); }
.ks-trait .ex { color:var(--muted); font-size:12.5px; margin-top:6px; line-height:1.8; }
/* player cards (alternatives / search results) */
.ks-pl { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:14px; box-shadow:var(--shadow); display:flex; gap:12px; min-height:132px; }
.ks-pl img { width:64px; height:64px; object-fit:contain; object-position:center bottom; border-radius:50%;
  background:radial-gradient(circle at 50% 35%, #E8EEF6 0%, #D5DEEA 100%); }
.ks-pl .nm { font-weight:800; color:var(--ink); } .ks-pl .meta { color:var(--muted); font-size:12.5px; }
.ks-pl .why { color:#15803D; font-size:12.5px; margin-top:4px; line-height:1.7; }
.ks-pl .rank { font-family:'Tajawal'; font-weight:900; color:var(--muted); font-size:14px; }
.ks-match { font-family:'Tajawal'; font-weight:900; color:#fff; background:var(--accent); border-radius:10px; padding:2px 9px; font-size:14px; white-space:nowrap; }
.ks-price { font-size:12.5px; color:var(--ink2); margin-top:3px; } .ks-price b { color:var(--ink); }
/* timeline / tables / requirements */
.ks-tl { border-inline-start:2px solid var(--line); margin-inline-start:8px; padding-inline-start:18px; }
.ks-tl .it { position:relative; margin:0 0 16px; } .ks-tl .it:before { content:''; position:absolute; inset-inline-start:-25px; top:7px;
  width:12px; height:12px; border-radius:50%; background:#0EA5E9; box-shadow:0 0 0 4px #fff; }
.ks-tl .pd { font-weight:800; color:var(--ink); } .ks-tl .ph { font-size:12px; padding:1px 9px; border-radius:999px; background:var(--soft); margin-inline-start:6px; color:var(--ink2); }
.ks-tl .it div { color:var(--ink2); line-height:1.85; }
.ks-tag { display:inline-block; padding:1px 8px; margin:2px; border-radius:8px; background:var(--soft); font-size:12px; color:var(--ink2); }
table.ks-t { width:100%; border-collapse:collapse; font-size:13.5px; } table.ks-t th { color:var(--muted); text-align:right; font-weight:600; padding:8px 6px; border-bottom:1px solid var(--line); }
table.ks-t td { border-bottom:1px solid var(--line); padding:8px 6px; color:var(--ink2); }
.ks-req { display:grid; grid-template-columns:1.3fr 90px 2fr; gap:12px; align-items:center; padding:10px 0; border-top:1px solid var(--line); }
.ks-req:first-child { border-top:0; }
/* بطاقة كشّاف (stays dark - the signature object) */
.kc { position:relative; max-width:320px; margin:0 auto; border-radius:24px; overflow:hidden; color:#E6EAF2;
  background:linear-gradient(180deg,#141E33 0%,#0B1222 100%);
  box-shadow:0 22px 44px rgba(11,18,34,.35); clip-path:polygon(0 0, 84% 0, 100% 8%, 100% 100%, 16% 100%, 0 92%); }
.kc:before { content:''; position:absolute; inset:0; pointer-events:none; background:repeating-linear-gradient(135deg, rgba(255,255,255,.028) 0 2px, transparent 2px 13px); }
.kc-in { position:relative; padding:18px 18px 16px; }
.kc-top { display:flex; justify-content:space-between; align-items:center; font-size:12px; }
.kc-brand { font-family:'Tajawal'; font-weight:900; opacity:.85; } .kc-tier { padding:3px 10px; border-radius:999px; font-weight:800; color:#0A0F1C; }
.kc-pos { padding:3px 10px; border-radius:999px; background:rgba(255,255,255,.1); }
.kc-ph { position:relative; width:158px; height:158px; margin:14px auto 8px; border-radius:50%; padding:5px; background:rgba(255,255,255,.08); }
.kc-disc { width:100%; height:100%; border-radius:50%; overflow:hidden; border:3px solid #0B1222; display:flex; align-items:flex-end; justify-content:center; }
.kc-disc img { width:100%; height:100%; object-fit:cover; object-position:center top; }
.kc-disc img.cut { width:94%; height:94%; object-fit:contain; object-position:center bottom; filter:drop-shadow(0 6px 10px rgba(0,0,0,.35)); }
.kc-logo { position:absolute; top:6px; right:2px; width:40px; height:40px; border-radius:50%; background:#fff; padding:5px;
  box-shadow:0 4px 12px rgba(0,0,0,.35); }
.kc-logo img { width:100%; height:100%; object-fit:contain; }
.kc-ovr { position:absolute; bottom:-10px; left:50%; transform:translateX(-50%); background:#0B1222; border:2px solid #16A34A;
  border-radius:12px; padding:0 12px; font-family:'Tajawal'; font-weight:900; font-size:25px; line-height:1.35; color:#fff; }
.kc-name { text-align:center; font-family:'Tajawal'; font-weight:900; font-size:22px; margin-top:16px; color:#fff; line-height:1.25; }
.kc-sub { text-align:center; color:#AEB8CB; font-size:12.5px; margin:2px 0 4px; }
.kc-dna { display:flex; flex-wrap:wrap; justify-content:center; gap:5px; margin:6px 0 8px; }
.kc-dna span { font-size:11.5px; padding:2px 9px; border-radius:999px; background:rgba(34,197,94,.14); border:1px solid rgba(34,197,94,.4); color:#BBF7D0; }
.kc-dna span.neg { background:rgba(239,68,68,.12); border-color:rgba(239,68,68,.4); color:#FECACA; }
.kc-foot { display:grid; grid-template-columns:repeat(3,1fr); border-top:1px solid rgba(255,255,255,.1); padding-top:8px; text-align:center; }
.kc-foot .k { font-size:10.5px; color:#8C97AD; } .kc-foot .v { font-family:'Tajawal'; font-weight:800; font-size:15px; color:#fff; }
.ks-empty { text-align:center; padding:50px 20px; } .ks-empty .t { font-family:'Tajawal'; font-weight:900; font-size:26px; color:var(--ink); }
@media (max-width:900px) { .ks-kpis { grid-template-columns:repeat(2,1fr); } .ks-hero .nm { font-size:28px; } }
@media (max-width:760px) { .ks-attr { grid-template-columns:110px 1fr 40px; } .ks-req { grid-template-columns:1fr; } }
</style>
"""

PLOT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans Arabic", color="#334155"), margin=dict(t=30, b=10, l=10, r=10))
GRID = dict(gridcolor="#E5E9F0", zerolinecolor="#E5E9F0")
PHASE_COLOR = {"بداية": "#94A3B8", "صعود": "#16A34A", "ذروة": "#C99A2E", "ثبات": "#0EA5E9", "تراجع": "#DC2626",
               "انتعاش": "#65A30D", "إصابة": "#EA580C", "انتقال": "#7C3AED"}
VERDICT_COLOR = {"يناسب": "#16A34A", "يناسب جزئيًا": "#CA8A04", "لا يناسب": "#DC2626"}
LEVEL_COLOR = {"قوي": "#16A34A", "مقبول": "#CA8A04", "ضعيف": "#DC2626"}
TRAIT_SHORT = {"physical_effort": "المجهود البدني", "ball_carrying": "التقدم بالكرة", "press_resistance": "الاحتفاظ بالكرة",
               "progression": "الكرات الطويلة", "chance_creation": "صناعة الفرص", "end_product": "الحسم",
               "tactical_intelligence": "ذكاء تكتيكي", "open_space": "المساحات المفتوحة", "tight_space": "المساحات الضيقة",
               "pressing": "الضغط على الخصم", "duels": "الالتحامات", "defensive_discipline": "الانضباط الدفاعي",
               "consistency": "الثبات"}


def flat(html_text: str) -> str:
    """Collapse multi-line HTML: Streamlit's markdown turns indented lines after a blank line into code blocks."""
    return "".join(line.strip() for line in html_text.splitlines())


_FORMATION = re.compile(r"(?<![\d.])\d(?:-\d){1,4}(?![\d.])")


def esc(x) -> str:
    """HTML-escape, and keep formations like 3-5-2 left-to-right inside Arabic text (else they show as 2-5-3)."""
    text = html.escape(str(x if x is not None else ""))
    return _FORMATION.sub(lambda m: f"<span dir='ltr' style='unicode-bidi:isolate'>{m.group(0)}</span>", text)


def eur(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if v >= 1e6:
        return f"€{v / 1e6:,.1f}M"
    if v >= 1e3:
        return f"€{v / 1e3:,.0f}K"
    return f"€{v:,.0f}"


def rcolor(v) -> str:
    if v is None:
        return "#94A3B8"
    return "#15803D" if v >= 85 else "#16A34A" if v >= 75 else "#CA8A04" if v >= 65 else "#EA580C" if v >= 55 else "#DC2626"


def tcolor(v) -> str:
    if v is None:
        return "#94A3B8"
    return "#15803D" if v >= 80 else "#16A34A" if v >= 65 else "#CA8A04" if v >= 50 else "#EA580C" if v >= 40 else "#DC2626"


def tier(o) -> tuple[str, str]:
    o = o or 0
    if o >= 86:
        return "نخبة عالمية", "#E0B341"
    if o >= 80:
        return "نخبة", "#22C55E"
    if o >= 74:
        return "مميز", "#38BDF8"
    if o >= 68:
        return "أساسي", "#A3E635"
    if o >= 60:
        return "متوسط", "#EAB308"
    return "بديل", "#F97316"


def ring(value, label="", color=None, pct=None) -> str:
    p = pct if pct is not None else (value or 0)
    return (f"<div class='ks-ring' style='background:conic-gradient({color or tcolor(value)} {p or 0}%, #F1F4F9 0)'><div><div>"
            f"<div class='v'>{value if value is not None else '—'}</div><div class='l'>{esc(label)}</div></div></div></div>")


def tile(label, value, sub="") -> str:
    return f"<div class='ks-tile'><div class='l'>{esc(label)}</div><div class='v'>{esc(value)}</div><div class='s'>{esc(sub)}</div></div>"


def kpi(label, value, sub="") -> str:
    return f"<div class='ks-kpi'><div class='l'>{esc(label)}</div><div class='v'>{esc(value)}</div><div class='s'>{esc(sub)}</div></div>"


def section(title: str, sub: str = "") -> str:
    return (f"<div class='ks-sec'><span class='dot'></span><h3>{esc(title)}</h3>"
            f"{f'<small>{esc(sub)}</small>' if sub else ''}</div>")


def bar_row(label: str, value, color_fn=tcolor, right: str | None = None, cols: str = "170px 1fr 48px") -> str:
    if right is not None:
        right_html = f"<div class='ks-muted'>{esc(right)}</div>"
    else:
        right_html = f"<div class='ks-val' style='background:{color_fn(value)}'>{value if value is not None else '—'}</div>"
    return (f"<div class='ks-attr' style='grid-template-columns:{cols}'><div class='lb'>{esc(label)}</div>"
            f"<div class='ks-bar'><i style='width:{value or 0:.0f}%;background:{color_fn(value)}'></i></div>{right_html}</div>")


def pc_item(title: str, detail: str, score: int | None, positive: bool) -> str:
    k = "#16A34A" if positive else "#DC2626"
    pill = f"<span class='ks-score' style='background:{tcolor(score)}'>{score}</span>" if score is not None else ""
    return (f"<div class='ks-pc' style='border-right:4px solid {k}'><div class='top'><span class='tt'>{esc(title)}</span>{pill}</div>"
            f"<div class='dd'>{esc(detail)}</div></div>")


def trait_card(t: dict) -> str:
    data = f"البيانات: أفضل من {t['data_pct']}% من لاعبي مركزه" if t.get("data_pct") is not None else ""
    return (f"<div class='ks-trait'><div class='top'><span class='nm'>{esc(t['label'])}</span>"
            f"<span class='ks-score' style='background:{tcolor(t['score'])}'>{t['score']}</span></div>"
            f"<div class='ks-bar' style='margin-top:8px'><i style='width:{t['score']}%;background:{tcolor(t['score'])}'></i></div>"
            f"<div class='ex'>{esc(t.get('explanation') or data)}</div></div>")


def player_card(r, badge: str, lines: list[str], why: str = "", rank: int | None = None) -> str:
    fid = getattr(r, "fotmob_id", None)
    img = f"https://images.fotmob.com/image_resources/playerimages/{int(fid)}.png" if fid == fid and fid else         (esc(r.image_url) if isinstance(getattr(r, "image_url", None), str) else "")
    age = "—" if r.age is None or r.age != r.age else f"{r.age:.0f}"
    meta = (f"{esc(r.team)} · {esc(LEAGUE_NAMES_AR.get(r.league, r.league))} · {age} سنة · "
            f"{esc(POSITION_NAMES_AR.get(r.pos_group, ''))}")
    rank_html = f"<span class='rank'>#{rank}</span> " if rank else ""
    price_lines = "".join(f"<div class='ks-price'>{ln}</div>" for ln in lines)
    why_html = f"<div class='why'>{esc(why)}</div>" if why else ""
    return (f"<div class='ks-pl'><img src='{img}'><div style='flex:1'>"
            f"<div style='display:flex;justify-content:space-between;gap:8px;align-items:center'>"
            f"<span class='nm'>{rank_html}{esc(r.name)}</span><span class='ks-match'>{esc(badge)}</span></div>"
            f"<div class='meta'>{meta}</div>{price_lines}{why_html}</div></div>")


def hex_radar(pillars: dict[str, int | None], size: int = 230) -> str:
    names = list(pillars)
    n = len(names)
    cx = cy = 115
    r = 68

    def pt(i, frac):
        a = math.radians(-90 + 360 / n * i)
        return cx + r * frac * math.cos(a), cy + r * frac * math.sin(a)

    grid = ""
    for frac in (1 / 3, 2 / 3, 1):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, frac) for i in range(n)))
        grid += f"<polygon points='{pts}' fill='none' stroke='rgba(255,255,255,.12)' stroke-width='1'/>"
    axes = "".join(f"<line x1='{cx}' y1='{cy}' x2='{pt(i, 1)[0]:.1f}' y2='{pt(i, 1)[1]:.1f}' stroke='rgba(255,255,255,.08)'/>"
                   for i in range(n))
    vals = [(pillars[k] or 0) / 100 for k in names]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, max(v, .04)) for i, v in enumerate(vals)))
    labels = ""
    for i, k in enumerate(names):
        x, y = pt(i, 1.3)
        v = pillars[k]
        labels += (f"<text x='{x:.1f}' y='{y - 2:.1f}' text-anchor='middle' font-size='11.5' fill='#C3CBDA'>{esc(k)}</text>"
                   f"<text x='{x:.1f}' y='{y + 12:.1f}' text-anchor='middle' font-size='13' font-weight='800' "
                   f"fill='{tcolor(v)}'>{v if v is not None else '—'}</text>")
    return (f"<svg viewBox='0 0 230 230' style='width:100%;max-width:{size}px;display:block;margin:auto'>{grid}{axes}"
            f"<polygon points='{poly}' fill='rgba(34,197,94,.28)' stroke='#22C55E' stroke-width='2'/>{labels}</svg>")


def kashaf_card(p: dict, pos_group: str, comb: dict, value: float | None, consistency: dict, club_color: str | None,
                photo: str | None = None, logo: str | None = None) -> str:
    o = comb["overall"]
    tname, tcol = tier(o)
    traits = [t for t in comb["traits"] if t["key"] != "consistency"]
    pos_tr = [t for t in traits if t["verdict"] == "قوة"][:3]
    neg_tr = [t for t in sorted(traits, key=lambda t: t["score"]) if t["verdict"] == "ضعف"][:1]
    dna = "".join(f"<span>{esc(TRAIT_SHORT.get(t['key'], t['label']))}</span>" for t in pos_tr)
    dna += "".join(f"<span class='neg'>{esc(TRAIT_SHORT.get(t['key'], t['label']))}</span>" for t in neg_tr)
    age = "—" if p.get("age") is None else f"{p['age']:.0f}"
    pot = comb.get("potential")
    club = club_color or "#1E3A8A"
    bg = (f"background:radial-gradient(130% 70% at 100% 0%, {club}88 0%, transparent 58%), "
          f"linear-gradient(180deg,#141E33 0%,#0B1222 100%)")
    photo = photo or p.get("image_url") or ""
    cut = "cut" if photo.endswith(".png") else ""
    disc = f"background:radial-gradient(circle at 50% 30%, {club}DD 0%, {club}55 45%, #0B1222 85%)"
    logo_html = f"<div class='kc-logo'><img src='{esc(logo)}'></div>" if logo else ""
    return flat(f"""<div class='kc' style="{bg}"><div class='kc-in'>
      <div class='kc-top'><span class='kc-brand'>كشّاف</span><span class='kc-pos'>{esc(POSITION_NAMES_AR.get(pos_group, pos_group))}</span>
        <span class='kc-tier' style='background:{tcol}'>{tname}</span></div>
      <div class='kc-ph' style='background:conic-gradient({tcolor(o)} {o or 0}%, rgba(255,255,255,.08) 0)'>
        <div class='kc-disc' style="{disc}"><img class='{cut}' src='{esc(photo)}'></div>{logo_html}
        <div class='kc-ovr' style='border-color:{tcolor(o)}'>{o if o is not None else '—'}</div></div>
      <div class='kc-name'>{esc(p['name'])}</div><div class='kc-sub'>{esc(p.get('current_club_name'))} · {age} سنة</div>
      {hex_radar(comb['pillars'])}<div class='kc-dna'>{dna}</div>
      <div class='kc-foot'><div><div class='k'>القيمة العادلة</div><div class='v'>{eur(value)}</div></div>
        <div><div class='k'>الثبات</div><div class='v'>{consistency['score'] if consistency['score'] is not None else '—'}</div></div>
        <div><div class='k'>الإمكانات</div><div class='v'>{pot if pot else '—'}</div></div></div></div></div>""")


PITCH_XY = {"GK": [(50, 92)], "CB": [(50, 79)], "FB": [(14, 72), (86, 72)], "DM": [(50, 64)], "CM": [(50, 51)],
            "AM": [(50, 37)], "W": [(15, 27), (85, 27)], "ST": [(50, 13)]}


def pitch(positions: list[dict]) -> str:
    dots = ""
    for p in positions:
        for x, y in PITCH_XY[p["group"]]:
            fill = "#C99A2E" if p["natural"] else "#FFFFFF"
            stroke = "#C99A2E" if p["natural"] else "#16A34A"
            games = "" if p.get("games") is None else str(p["games"])
            dots += (f"<circle cx='{x}' cy='{y}' r='6.2' fill='{fill}' stroke='{stroke}' stroke-width='.9'/>"
                     f"<text x='{x}' y='{y + 1.6}' text-anchor='middle' font-size='4.6' font-weight='800' "
                     f"fill='{'#fff' if p['natural'] else '#15803D'}'>{games}</text>"
                     f"<text x='{x}' y='{y + 10.5}' text-anchor='middle' font-size='3.6' fill='#E6F4EA'>{POSITION_NAMES_AR[p['group']]}</text>")
    return flat(f"""<svg viewBox='0 0 100 100' style='width:100%;max-width:320px;display:block;margin:auto'>
      <rect x='1' y='1' width='98' height='98' rx='5' fill='#14532D'/>
      <line x1='1' y1='50' x2='99' y2='50' stroke='#3F7D58' stroke-width='.6'/>
      <circle cx='50' cy='50' r='10' fill='none' stroke='#3F7D58' stroke-width='.6'/>
      <rect x='28' y='1' width='44' height='15' fill='none' stroke='#3F7D58' stroke-width='.6'/>
      <rect x='28' y='84' width='44' height='15' fill='none' stroke='#3F7D58' stroke-width='.6'/>{dots}</svg>""")
