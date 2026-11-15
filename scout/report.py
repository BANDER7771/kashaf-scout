"""PDF scouting report: print-designed HTML rendered by headless Edge/Chrome (correct Arabic shaping, real CSS)."""
from __future__ import annotations

import hashlib
import os
import html
import re
import subprocess
from pathlib import Path

import pandas as pd

from .config import CACHE_DIR, LEAGUE_NAMES_AR, POSITION_NAMES_AR, ROOT

BROWSERS = [
    os.environ.get("CHROME_PATH", ""),
    "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


class ReportError(RuntimeError):
    pass


_FORMATION = re.compile(r"(?<![\d.])\d(?:-\d){1,4}(?![\d.])")


def _e(x) -> str:
    """HTML-escape, keeping formations like 3-5-2 left-to-right inside Arabic text."""
    text = html.escape(str(x if x is not None else ""))
    return _FORMATION.sub(lambda m: f"<span dir='ltr' style='unicode-bidi:isolate'>{m.group(0)}</span>", text)


def _eur(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"€{v / 1e6:,.1f}M" if v >= 1e6 else f"€{v / 1e3:,.0f}K" if v >= 1e3 else f"€{v:,.0f}"


def _tc(v) -> str:
    if v is None:
        return "#94A3B8"
    return "#15803D" if v >= 80 else "#22C55E" if v >= 65 else "#CA8A04" if v >= 50 else "#EA580C" if v >= 40 else "#DC2626"


CSS = """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Tajawal:wght@500;700;800;900&display=swap" rel="stylesheet">
<style>
@page { size:A4; margin:12mm 12mm 14mm; }
* { box-sizing:border-box; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
body { direction:rtl; font-family:'IBM Plex Sans Arabic','Segoe UI',Tahoma,sans-serif; color:#0F172A; font-size:11.5px; line-height:1.75; margin:0; }
h1,h2,h3,.t { font-family:'Tajawal','Segoe UI',sans-serif; }
.hero { background:linear-gradient(120deg,#0B1222 0%,#13203A 60%,var(--club) 160%); color:#E6EAF2; border-radius:16px; padding:18px 20px;
  display:flex; gap:18px; align-items:center; }
.hero img { width:92px; height:108px; object-fit:cover; object-position:top; border-radius:12px; border:2px solid rgba(255,255,255,.2); }
.hero .nm { font-family:'Tajawal'; font-size:26px; font-weight:900; line-height:1.2; }
.hero .sub { color:#AEB8CB; font-size:12px; margin-top:2px; }
.hero .arch { color:#86EFAC; font-weight:700; font-size:14px; margin-top:6px; }
.ring { width:84px; height:84px; border-radius:50%; display:grid; place-items:center; margin-inline-start:auto;
  background:conic-gradient(var(--c) calc(var(--p)*1%), rgba(255,255,255,.12) 0); }
.ring div { width:66px; height:66px; border-radius:50%; background:#0B1222; display:grid; place-items:center; text-align:center; }
.ring b { font-family:'Tajawal'; font-size:26px; font-weight:900; line-height:1; display:block; } .ring span { font-size:9px; color:#AEB8CB; }
.brand { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; color:#475569; font-size:10.5px; }
.brand .lg { font-family:'Tajawal'; font-weight:900; font-size:18px; color:#0F172A; }
.brand .lg i { font-style:normal; color:#16A34A; }
.tiles { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin:12px 0; }
.tile { border:1px solid #E2E8F0; border-radius:10px; padding:7px 9px; background:#F8FAFC; }
.tile .l { color:#64748B; font-size:9.5px; } .tile .v { font-family:'Tajawal'; font-weight:800; font-size:14px; } .tile .s { color:#64748B; font-size:9px; }
.sec { margin-top:14px; page-break-inside:avoid; }
.sec h2 { font-size:15px; margin:0 0 8px; padding-inline-start:9px; border-inline-start:4px solid #16A34A; }
.cols { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.item { border:1px solid #E2E8F0; border-radius:10px; padding:7px 10px; margin-bottom:6px; page-break-inside:avoid; }
.item .h { display:flex; justify-content:space-between; gap:8px; font-weight:700; }
.item .d { color:#334155; font-size:10.5px; }
.pill { font-family:'Tajawal'; font-weight:900; font-size:11px; color:#fff; border-radius:6px; padding:0 7px; }
.pros .item { border-inline-start:3px solid #22C55E; } .cons .item { border-inline-start:3px solid #EF4444; }
.verdict { background:#F0FDF4; border:1px solid #BBF7D0; border-radius:12px; padding:10px 12px; }
.badge { display:inline-block; border-radius:999px; padding:1px 10px; font-weight:800; color:#fff; font-size:11px; }
table { width:100%; border-collapse:collapse; font-size:10.5px; } th { color:#64748B; text-align:right; font-weight:600; padding:4px 6px; border-bottom:1px solid #CBD5E1; }
td { padding:4px 6px; border-bottom:1px solid #E2E8F0; }
.bar { height:6px; background:#E2E8F0; border-radius:4px; overflow:hidden; } .bar i { display:block; height:100%; border-radius:4px; }
.muted { color:#64748B; font-size:10px; }
.page { page-break-before:always; }
ul { margin:4px 0; padding-inline-start:18px; } li { margin:2px 0; }
.foot { margin-top:16px; color:#94A3B8; font-size:9px; text-align:center; }
</style>
"""


def consistent_score(scores: dict, trait: str | None, positive: bool) -> int | None:
    """Show a trait score next to a pro/con only when it agrees with it (no '29' next to a strength)."""
    sc = scores.get(trait)
    if sc is None:
        return None
    return sc if (sc >= 50 if positive else sc <= 60) else None


def _items(rows: list[dict], scores: dict, positive: bool) -> str:
    out = ""
    for r in rows:
        sc = consistent_score(scores, r.get("trait"), positive)
        pill = f"<span class='pill' style='background:{_tc(sc)}'>{sc}</span>" if sc is not None else ""
        out += f"<div class='item'><div class='h'><span>{_e(r['title'])}</span>{pill}</div><div class='d'>{_e(r['detail'])}</div></div>"
    return out


def build_html(base: dict, comb: dict, syn: dict | None, alts: pd.DataFrame | None, full: bool) -> str:
    p, v, cons = base["player"], base["valuation"], base["consistency"]
    syn = syn or {}
    sm = syn.get("summary") or {}
    scores = {t["key"]: t["score"] for t in comb["traits"]}
    league = LEAGUE_NAMES_AR.get(base["league_code"], base["league_code"])
    o = comb["overall"]
    contract = "—" if pd.isna(p["contract_expiration_date"]) else f"{p['contract_expiration_date']:%Y-%m}"
    form = base["fotmob_matches"]
    form_txt = "—" if form.empty else f"{form.tail(10)['rating'].mean():.2f}"
    logo = (ROOT / "assets" / "logo.svg").read_text(encoding="utf-8").replace('width="64" height="64"', 'width="28" height="28"')
    parts = [f"""<div class='brand'><span class='lg' style='display:flex;align-items:center;gap:8px'>{logo}كشّاف</span><span>تقرير كشفي · {pd.Timestamp.now():%Y-%m-%d}</span></div>
    <div class='hero' style='--club:{base.get("team_color") or "#1E3A8A"}'>
      <img src='{_e(base.get("photo") or p.get("image_url"))}' style='{"object-fit:contain;background:radial-gradient(circle at 50% 30%,rgba(255,255,255,.25),rgba(255,255,255,.05))" if str(base.get("photo") or "").endswith(".png") else ""}'>
      <div><div class='nm'>{_e(p['name'])}</div>
        <div class='sub'>{_e(p['current_club_name'])} · {_e(league)} · {_e(POSITION_NAMES_AR.get(base['pos_group']))} · {p['age']:.0f} سنة · {_e(p['country_of_citizenship'])} · العقد حتى {contract}</div>
        <div class='arch'>{_e(sm.get('archetype'))}</div>
        <div class='sub' style='color:#CBD5E1'>{_e(sm.get('headline'))}</div></div>
      <div class='ring' style='--p:{o or 0};--c:{_tc(o)}'><div><span><b>{o if o is not None else '—'}</b>التقييم العام</span></div></div>
    </div>
    <div class='tiles'>
      <div class='tile'><div class='l'>القيمة العادلة</div><div class='v'>{_eur(v['base_value'])}</div></div>
      <div class='tile'><div class='l'>سعر الشراء العادل</div><div class='v'>{_eur(v['fee_low'])} - {_eur(v['fee_high'])}</div></div>
      <div class='tile'><div class='l'>الراتب السنوي العادل</div><div class='v'>{_eur(v['salary_low'])} - {_eur(v['salary_high'])}</div></div>
      <div class='tile'><div class='l'>ثبات المستوى</div><div class='v'>{cons['score'] if cons['score'] is not None else '—'}/100</div><div class='s'>{_e(cons['label'])}</div></div>
      <div class='tile'><div class='l'>آخر 10 مباريات</div><div class='v'>{form_txt}</div><div class='s'>متوسط التقييم</div></div>
    </div>"""]
    if syn.get("verdict"):
        parts.append(f"<div class='sec verdict'><b>الخلاصة:</b> {_e(syn['verdict'])}</div>")
    pros, cons_ = syn.get("pros") or [], syn.get("cons") or []
    if pros or cons_:
        parts.append(f"<div class='sec cols'><div class='pros'><h2>أبرز الإيجابيات</h2>{_items(pros, scores, True)}</div>"
                     f"<div class='cons'><h2>أبرز السلبيات</h2>{_items(cons_, scores, False)}</div></div>")
    if syn.get("suitable_teams"):
        st_ = "".join(f"<div class='item'><div class='h'>{_e(t['profile'])}</div><div class='d'>{_e(t['why'])}</div></div>" for t in syn["suitable_teams"])
        un_ = "".join(f"<div class='item'><div class='h'>{_e(t['profile'])}</div><div class='d'>{_e(t['why'])}</div></div>" for t in syn.get("unsuitable_teams") or [])
        parts.append(f"<div class='sec cols'><div><h2>نوعية الفرق التي تناسبه</h2>{st_}</div><div><h2>فرق لا تناسبه</h2>{un_}</div></div>")
    cf = syn.get("coach_fit")
    if cf:
        col = {"يناسب": "#16A34A", "يناسب جزئيًا": "#CA8A04", "لا يناسب": "#DC2626"}.get(cf["verdict"], "#CA8A04")
        methods = "".join(f"<li><b>{_e(m['principle'])}:</b> {_e(m['detail'])}</li>" for m in cf["coach_methods"])
        parts.append(f"""<div class='sec'><h2>أسلوب {_e(cf['coach_name'])} ومدى توافقه <span class='badge' style='background:{col}'>{_e(cf['verdict'])}</span></h2>
          <div class='muted'>{_e(cf['club_name'])} · {_e(cf['formation'])}</div>
          <div class='cols'><div class='item'><div class='h'>أسلوب المدرب</div><ul>{methods}</ul></div>
          <div class='item'><div class='h'>التحليل</div><div class='d'>{_e(cf['fit_analysis'])}</div></div></div></div>""")
    if alts is not None and not alts.empty:
        rows = "".join(f"<tr><td><b>{_e(a.name)}</b></td><td>{_e(a.team)}</td><td>{'' if pd.isna(a.age) else f'{a.age:.0f}'}</td>"
                       f"<td>{_eur(a.market_value_in_eur)}</td><td>{a.rating:.2f}</td><td>{_e(a.why)}</td></tr>" for a in alts.head(5).itertuples())
        parts.append(f"<div class='sec'><h2>بدلاء أفضل قيمة فنية ومالية</h2><table><tr><th>اللاعب</th><th>النادي</th><th>العمر</th>"
                     f"<th>القيمة</th><th>التقييم</th><th>لماذا</th></tr>{rows}</table></div>")
    if syn.get("risks"):
        parts.append("<div class='sec'><h2>المخاطر</h2><ul>" + "".join(f"<li>{_e(r)}</li>" for r in syn["risks"]) + "</ul></div>")

    if full:
        parts.append("<div class='page'></div>")
        if syn.get("playing_style"):
            parts.append(f"<div class='sec'><h2>أسلوب اللعب</h2><p>{_e(syn['playing_style'])}</p></div>"
                         f"<div class='sec'><h2>الدور والمنظومة الأنسب</h2><p>{_e(syn.get('position_and_system'))}</p></div>")
        trait_rows = "".join(f"<tr><td>{_e(t['label'])}</td><td style='width:38%'><div class='bar'><i style='width:{t['score']}%;background:{_tc(t['score'])}'></i></div></td>"
                             f"<td><b>{t['score']}</b></td><td class='muted'>{_e(t.get('explanation') or t.get('basis'))}</td></tr>" for t in comb["traits"])
        parts.append(f"<div class='sec'><h2>سمات اللعب <span class='muted'>(50 = متوسط مركزه)</span></h2><table>{trait_rows}</table></div>")
        if base["peer_metrics"]:
            pm = ""
            for m in base["peer_metrics"]:
                rank = "—" if m["pct"] is None or pd.isna(m["pct"]) else f"أفضل من {float(m['pct']):.0f}%"
                pm += f"<tr><td>{_e(m['metric'])}</td><td>{m['value']:.2f}</td><td>{rank}</td></tr>"
            parts.append(f"<div class='sec'><h2>الأرقام مقارنة بلاعبي مركزه <span class='muted'>{_e(base['rating_season'])}</span></h2>"
                         f"<table><tr><th>المؤشر</th><th>لكل 90 دقيقة</th><th>الترتيب</th></tr>{pm}</table></div>")
        if cf:
            reqs = "".join(f"<tr><td>{_e(r['requirement'])}</td><td><b>{_e(r['player_level'])}</b></td><td>{_e(r['note'])}</td></tr>"
                           for r in cf["requirements"])
            parts.append(f"<div class='sec'><h2>ما يطلبه {_e(cf['coach_name'])} مقابل مستوى اللاعب ({cf['fit_score']}%)</h2>"
                         f"<div class='muted'>الدور المتوقع: {_e(cf['role_for_player'])}</div>"
                         f"<table><tr><th>ما يطلبه المدرب</th><th>مستوى اللاعب</th><th>الدليل</th></tr>{reqs}</table>"
                         + (f"<p><b>هوية النادي:</b> {_e(cf['club_identity'])}</p>" if cf.get("club_identity") else "")
                         + (f"<p><b>أهداف النادي:</b> {_e(cf['club_objectives'])}</p>" if cf.get("club_objectives") else "") + "</div>")
        seasons = "".join(f"<tr><td>{_e(s['label'])}</td><td>{_e(s['club'])}</td><td>{s['games']}</td><td>{s['goals']}</td><td>{s['assists']}</td>"
                          f"<td>{'—' if s['rating'] is None else round(s['rating'], 2)}</td><td>{_eur(s['market_value_end'])}</td>"
                          f"<td class='muted'>{_e(' · '.join(s['tags']))}</td></tr>" for s in reversed(base["seasons"]))
        parts.append(f"<div class='sec'><h2>المسيرة موسمًا بموسم</h2><table><tr><th>الموسم</th><th>النادي</th><th>مباريات</th><th>أهداف</th>"
                     f"<th>صناعة</th><th>التقييم</th><th>القيمة</th><th>مؤشرات</th></tr>{seasons}</table></div>")
        if syn.get("career_phases"):
            parts.append("<div class='sec'><h2>مراحل المسيرة وأسبابها</h2>" + "".join(
                f"<div class='item'><div class='h'>{_e(ph['period'])} · {_e(ph['phase'])}</div><div class='d'>{_e(ph['explanation'])}</div></div>"
                for ph in syn["career_phases"]) + "</div>")
        if syn.get("decline_analysis"):
            parts.append(f"<div class='sec'><h2>الثبات والهبوط</h2><p>{_e(syn.get('consistency_analysis'))}</p><p>{_e(syn['decline_analysis'])}</p></div>")
        parts.append(f"""<div class='sec'><h2>القيمة</h2><table>
          <tr><td>نموذجنا (الأداء والعمر والعقد)</td><td>{_eur(v['model_value'])}</td></tr>
          <tr><td>Transfermarkt</td><td>{_eur(v['tm_value'])}</td></tr><tr><td>SciSports</td><td>{_eur(v['scisports_value'])}</td></tr>
          <tr><td><b>القيمة العادلة</b></td><td><b>{_eur(v['base_value'])}</b></td></tr>
          <tr><td>سعر الشراء العادل</td><td>{_eur(v['fee_low'])} - {_eur(v['fee_high'])} ({_e(v['contract_note'])})</td></tr>
          <tr><td>الراتب السنوي العادل</td><td>{_eur(v['salary_low'])} - {_eur(v['salary_high'])}</td></tr></table>
          <p>{_e(syn.get('valuation_view'))}</p><p class='muted'>الراتب المُعلن: {_e(syn.get('reported_salary'))}</p></div>""")
        mk = syn.get("marketing") or {}
        if mk.get("analysis"):
            parts.append(f"<div class='sec'><h2>القيمة التسويقية والفنية <span class='badge' style='background:#0EA5E9'>{mk.get('score')}/100</span></h2>"
                         f"<p><b>{_e(mk.get('social_reach'))}</b></p><p>{_e(mk['analysis'])}</p><p>{_e(syn.get('technical_value'))}</p></div>")
        gaps = sm.get("data_gaps") or []
        if gaps:
            parts.append("<div class='sec'><h2>حدود التحليل</h2><ul>" + "".join(f"<li>{_e(g)}</li>" for g in gaps) + "</ul></div>")
    parts.append("<div class='foot'>كشّاف · تحليل موضوعي مبني على بيانات المباريات والقيم السوقية وتحليل المصادر · للاستخدام الخاص</div>")
    return f"<!doctype html><html lang='ar' dir='rtl'><head><meta charset='utf-8'>{CSS}</head><body>{''.join(parts)}</body></html>"


def to_pdf(doc: str) -> bytes:
    browser = next((b for b in BROWSERS if b and Path(b).exists()), None)
    if not browser:
        raise ReportError("لم يُعثر على Edge أو Chrome لإنشاء ملف PDF.")
    out_dir = CACHE_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(doc.encode("utf-8")).hexdigest()[:12]
    src, pdf = out_dir / f"{key}.html", out_dir / f"{key}.pdf"
    src.write_text(doc, encoding="utf-8")
    # A private profile keeps headless printing independent of any Edge/Chrome window the user has open.
    profile_dir = out_dir / "browser-profile"
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-first-run", "--disable-extensions", "--no-pdf-header-footer",
           "--virtual-time-budget=8000", f"--user-data-dir={profile_dir}", f"--print-to-pdf={pdf}", src.resolve().as_uri()]
    if os.name != "nt":  # containers run as a user without the kernel features Chromium's sandbox needs
        cmd[1:1] = ["--no-sandbox", "--disable-dev-shm-usage"]
    try:
        subprocess.run(cmd, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired as e:
        raise ReportError("استغرق إنشاء ملف PDF وقتًا أطول من المتوقع.") from e
    if not pdf.exists() or pdf.stat().st_size == 0:
        raise ReportError("تعذّر إنشاء ملف PDF.")
    return pdf.read_bytes()
