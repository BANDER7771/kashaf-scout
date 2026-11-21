"""كشّاف - منصة تحليل وتقييم اللاعبين (نسخة تجريبية خاصة)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from scout import ai, metrics, pool, profile, report, settings, spec, valuation, web
from scout import data_tm as tm
from scout.config import LEAGUE_NAMES_AR, POSITION_NAMES_AR, POOL_LEAGUES
from ui_kit import (CSS, GRID, LEVEL_COLOR, PHASE_COLOR, PLOT, VERDICT_COLOR, bar_row, esc, eur, flat, kashaf_card, kpi,
                    pc_item, pitch, player_card, rcolor, ring, section, tile, trait_card)

ASSETS = Path(__file__).parent / "assets"
LOGO_SVG = (ASSETS / "logo.svg").read_text(encoding="utf-8")
st.set_page_config(page_title="كشّاف | تحليل اللاعبين", page_icon=str(ASSETS / "favicon.png"), layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)

MODES = ["تحليل لاعب", "البحث بالمواصفات"]
PAGES = ["الملخص", "أسلوب اللعب", "المدرب والنادي", "المسيرة والثبات", "القيمة والبدلاء", "التقرير PDF", "المصادر"]


# ================================================================== state helpers

def go_to(page: str) -> None:
    st.session_state["page"] = page


def open_player(player_id: int) -> None:
    """Open any player (from alternatives or search results) in the analysis view."""
    st.session_state.update(mode=MODES[0], pid=int(player_id), notes=None, auto_ai=True, page=PAGES[0])


for _key, _options in (("mode", MODES), ("page", PAGES)):
    if st.session_state.get(_key) not in _options:
        st.session_state[_key] = _options[0]


# ================================================================== cached data

@st.cache_resource(show_spinner="تجهيز قواعد البيانات لأول مرة...")
def warm_up():
    tm.players()
    metrics.league_seasons()
    valuation.load_model()
    return True


@st.cache_resource(show_spinner="تجهيز قاعدة أساليب اللعب لـ 16 دوري (مرة أسبوعيًا)...")
def warm_pool():
    return len(pool.pool())


@st.cache_data(ttl=6 * 3600, max_entries=16, show_spinner="جمع بيانات المباريات والقيم السوقية...")
def get_base(player_id: int) -> dict:
    return profile.build_base(player_id)


@st.cache_data(ttl=24 * 3600, max_entries=32, show_spinner="البحث عن لاعبين بنفس الأسلوب...")
def get_alternatives(player_id: int, scope: str) -> pd.DataFrame:
    b = get_base(player_id)
    return pool.alternatives(b["pos_group"], b["rating_adj"], b["valuation"]["base_value"], float(b["player"]["age"] or 27),
                             b["fotmob_id"], profile.style_flat(b), scope=scope, league=b["league_code"])


@st.cache_data(max_entries=8, show_spinner="تجهيز ملف PDF...")
def get_pdf(player_id: int, club, coach, notes, saved_at, full: bool) -> bytes:
    b = get_base(player_id)
    syn_ = (ai.load_cached(player_id, club, coach, notes) or {}).get("synthesis")
    return report.to_pdf(report.build_html(b, profile.combine(b, syn_), syn_, get_alternatives(player_id, "all"), full))


# ---------------------------------------------------------------- access (server only: APP_PASSWORD secret)
_pw = settings.app_password()
if _pw and not st.session_state.get("authed"):
    st.markdown(flat(f"""<div class='ks-panel ks-empty' style='max-width:460px;margin:60px auto 16px'>
      <div style='width:64px;margin:0 auto 12px'>{LOGO_SVG}</div><div class='t'>كشّاف</div>
      <div class='ks-muted'>منصة تحليل وتقييم اللاعبين · نسخة تجريبية خاصة</div></div>"""), unsafe_allow_html=True)
    _c = st.columns([1, 1.4, 1])[1]
    _entered = _c.text_input("كلمة المرور", type="password")
    if _c.button("دخول", type="primary", use_container_width=True):
        if _entered == _pw:
            st.session_state["authed"] = True
            st.rerun()
        _c.error("كلمة المرور غير صحيحة.")
    st.stop()

warm_up()
ai_on = ai.available()
quota = settings.ai_quota_left()
if quota == 0:
    ai_on = False

# ================================================================== top bar + mode

st.markdown(flat(f"""<div class='ks-top'><div class='ks-logo'><div class='mark'>{LOGO_SVG}</div><div>
  <div class='name'>كشّاف</div><div class='tag'>منصة تحليل وتقييم اللاعبين</div></div></div>
  <div><span class='ks-chip {"ok" if ai_on else "off"}'>{"● الذكاء الاصطناعي متصل" if ai_on else "● الذكاء الاصطناعي غير مربوط"}</span>
  <span class='ks-chip'>بيانات حتى {tm.reference_date():%Y-%m}</span></div></div>"""), unsafe_allow_html=True)

if quota == 0:
    st.info("انتهى حد التحليلات الذكية لليوم - يمكنك تصفح التحليلات المحفوظة واستخدام البيانات، وسيتجدد الحد غدًا.")
elif not ai_on and not settings.hosted():
    with st.container(border=True):
        st.markdown("**ربط DeepSeek (مرة واحدة فقط)** - يُحفظ المفتاح في ملف `.env` على جهازك ويُستخدم تلقائيًا بعد ذلك.")
        k1, k2 = st.columns([4, 1])
        key = k1.text_input("مفتاح DeepSeek", type="password", label_visibility="collapsed", placeholder="sk-...")
        if k2.button("حفظ وربط", type="primary", use_container_width=True) and key.strip():
            settings.save_deepseek_key(key)
            st.rerun()

with st.sidebar:
    st.markdown("### الإعدادات")
    st.caption(f"الذكاء الاصطناعي: DeepSeek ({settings.deepseek_model()})")
    if quota is not None:
        st.caption(f"التحليلات المتبقية اليوم: {quota}")
    if ai_on and not settings.hosted():
        with st.expander("تغيير مفتاح DeepSeek"):
            new_key = st.text_input("المفتاح الجديد", type="password")
            if st.button("حفظ") and new_key.strip():
                settings.save_deepseek_key(new_key)
                st.rerun()

mode = st.radio("القسم", MODES, horizontal=True, key="mode", label_visibility="collapsed")


def player_grid(rows: pd.DataFrame, badge_fn, lines_fn, why_fn=lambda r: "", ranked: bool = False, key: str = "g") -> None:
    """Cards in a 3-column grid, each with a button that opens the player's full analysis."""
    items = list(rows.itertuples())
    for start in range(0, len(items), 3):
        cols = st.columns(3)
        for col, r, i in zip(cols, items[start:start + 3], range(start, start + 3)):
            with col:
                st.markdown(player_card(r, badge_fn(r), lines_fn(r), why_fn(r), i + 1 if ranked else None),
                            unsafe_allow_html=True)
                if pd.notna(getattr(r, "player_id", None)):
                    st.button("تحليل اللاعب ←", key=f"{key}_{r.fotmob_id}", on_click=open_player,
                              args=(int(r.player_id),), use_container_width=True)


# ================================================================== MODE: search by specification

def spec_mode() -> None:
    n_pool = warm_pool()
    st.markdown(section("ابحث عن لاعب بالمواصفات", f"{n_pool:,} لاعب أساسي في 16 دوري"), unsafe_allow_html=True)
    with st.container(border=True):
        q = st.text_area("صف اللاعب الذي تبحث عنه", key="spec_query", height=90,
                         placeholder="مثال: جناح أيسر يجيد المراوغة وصناعة الفرص، أقل من 25 سنة، قيمته أقل من 8 مليون، من الدوريات الأوروبية")
        st.caption("أمثلة: «لاعب وسط مجاني يجيد الكرات الطويلة من دوري روشن» · «قلب دفاع قوي في الالتحامات والكرات الهوائية أقل من 3 مليون» · "
                   "«مهاجم يضغط ويسجل، عمره أقل من 23»")
        c1, c2 = st.columns([1, 4])
        go_search = c1.button("ابحث", type="primary", use_container_width=True, disabled=not ai_on)
        if not ai_on:
            c2.caption("البحث بالوصف يحتاج DeepSeek - أو استخدم المعايير اليدوية بالأسفل.")
    if go_search and q.strip():
        with st.spinner("فهم الطلب..."):
            try:
                st.session_state["spec"] = spec.parse(q)
                st.session_state.pop("spec_reasons", None)
            except ai.AIError as e:
                st.error(str(e))
    sp = st.session_state.get("spec") or {"positions": [], "scope": "all", "style_weights": {}, "quality_weight": 0.35}
    if sp.get("understood_ar"):
        st.markdown(f"<div class='ks-verdict'><b>فهمت طلبك:</b> {esc(sp['understood_ar'])}</div>", unsafe_allow_html=True)

    with st.expander("المعايير (يمكنك تعديلها)", expanded=not ai_on):
        c1, c2, c3 = st.columns(3)
        positions = c1.multiselect("المركز", list(POSITION_NAMES_AR), default=sp.get("positions") or [],
                                   format_func=POSITION_NAMES_AR.get)
        ages = c2.slider("العمر", 16, 40, (int(sp.get("age_min") or 16), int(sp.get("age_max") or 40)))
        max_val = c3.number_input("أقصى قيمة سوقية (مليون €)", 0.0, 300.0, float((sp.get("max_value_eur") or 0) / 1e6), 0.5)
        c1, c2, c3 = st.columns(3)
        scopes = {"all": "كل الدوريات", "arab": "الدوريات العربية", "europe": "أوروبا"}
        scope = c1.selectbox("النطاق", list(scopes), index=list(scopes).index(sp.get("scope") or "all"), format_func=scopes.get)
        codes = [c for c, _ in POOL_LEAGUES.values()]
        leagues = c2.multiselect("دوريات محددة", codes, default=sp.get("leagues") or [],
                                 format_func=lambda c: LEAGUE_NAMES_AR.get(c, c))
        free = c3.checkbox("عقده ينتهي قريبًا (صفقة مجانية)", value=bool(sp.get("contract_expiring")))
        weights = sp.get("style_weights") or {}
        feats = c1.multiselect("المواصفات المطلوبة في أسلوبه", pool.STYLE_FEATURES + pool.PHYS_FEATURES,
                               default=[f for f in weights if f in pool.FEATURE_AR], format_func=pool.FEATURE_AR.get)
        quality = c2.slider("أهمية المستوى الحالي مقابل الأسلوب", 0.0, 1.0, float(sp.get("quality_weight") or 0.35), 0.05)
        apply_manual = st.button("تحديث النتائج")
    if not (apply_manual or "spec" in st.session_state):
        st.markdown("<div class='ks-panel ks-empty'><div class='t'>اكتب وصف اللاعب الذي تحتاجه</div>"
                    "<div class='ks-muted'>المركز، الأسلوب، العمر، الميزانية، الدوري، أو صفقة مجانية - "
                    "وتظهر النتائج مرتبة من الأنسب للأقل مع السعر والراتب المتوقعين.</div></div>", unsafe_allow_html=True)
        return
    sp = {**sp, "positions": positions, "age_min": ages[0] if ages[0] > 16 else None,
          "age_max": ages[1] if ages[1] < 40 else None, "max_value_eur": max_val * 1e6 if max_val else None,
          "scope": scope, "leagues": leagues, "contract_expiring": free, "quality_weight": quality,
          "style_weights": {f: weights.get(f, 1.0) for f in feats}}
    if apply_manual:
        st.session_state["spec"] = sp
        st.session_state.pop("spec_reasons", None)

    with st.spinner("ترتيب اللاعبين وحساب الأسعار..."):
        res = spec.run(sp, n=18)
    if res.empty:
        st.warning("لا يوجد لاعبون يطابقون كل الشروط - خفّف أحد المعايير (العمر أو الميزانية أو الدوري).")
        return
    if ai_on and st.session_state.get("spec_query") and "spec_reasons" not in st.session_state:
        with st.spinner("كتابة سبب ترشيح كل لاعب..."):
            try:
                st.session_state["spec_reasons"] = spec.explain(st.session_state["spec_query"], res.head(12), sp)
            except ai.AIError:
                st.session_state["spec_reasons"] = {}
    reasons = st.session_state.get("spec_reasons") or {}
    st.markdown(section("النتائج", "من الأنسب للأقل · الأسعار تقديرية وبيانات العقود حتى يونيو 2026"), unsafe_allow_html=True)

    def lines(r):
        fee = "<b>مجانًا</b>" if r.fee_high == 0 else f"<b dir='ltr'>{eur(r.fee_low)} - {eur(r.fee_high)}</b>"
        return [f"سعر الشراء المتوقع: {fee}", f"الراتب السنوي المتوقع: <b dir='ltr'>{eur(r.salary_low)} - {eur(r.salary_high)}</b>",
                f"{esc(r.status)} · التقييم <b>{r.rating:.2f}</b>", esc(spec.key_stats(r, sp))]
    player_grid(res, lambda r: f"مطابقة {r.match:.0f}%", lines, lambda r: reasons.get(str(r.fotmob_id), ""),
                ranked=True, key="spec")


# ================================================================== MODE: player analysis

def analysis_mode() -> None:
    pid = st.session_state.get("pid")
    with st.expander("البحث عن لاعب" if not pid else "تحليل لاعب آخر", expanded=not pid):
        c1, c2, c3 = st.columns([3, 2, 2])
        query = c1.text_input("اسم اللاعب", placeholder="سالم الدوسري أو Salem Al-Dawsari")
        target_club = c2.text_input("النادي المستهدف (اختياري)", placeholder="مثال: النصر")
        coach = c3.text_input("المدرب (اختياري)", placeholder="مثال: إنزاغي")
        c1, c2 = st.columns([3, 2])
        notes = c1.text_area("ملاحظاتك كمتابع للاعب (اختياري)", height=68,
                             placeholder="ما تراه بعينك ولا تقيسه البيانات، مثل الركض والشدة. تُدمج مع الأرقام كمشاهدة كشّاف.")
        langs = c2.multiselect("لغات بحث إضافية (تلقائي إذا تُركت فارغة)", list(web.LANGUAGES), format_func=web.LANGUAGES.get,
                               help="العربية والإنجليزية دائمًا. التلقائي يختار حسب جنسية اللاعب ودورياته.")
        if query:
            names = [query]
            if re.search(r"[؀-ۿ]", query):
                if ai_on:
                    try:
                        names = ai.transliterate(query) or [query]
                    except ai.AIError as e:
                        st.error(str(e))
                else:
                    st.info("للبحث بالعربي اربط DeepSeek أولًا، أو اكتب الاسم بالإنجليزية.")
            results = pd.concat([tm.search_players(n, limit=8) for n in names]).drop_duplicates("player_id")
            results = results.sort_values(["score", "market_value_in_eur"], ascending=False).head(10)
            if results.empty:
                st.warning("لم يُعثر على لاعب بهذا الاسم.")
            else:
                labels = {int(r.player_id): f"{r['name']} — {r.current_club_name} — {r.age:.0f} سنة — "
                                            f"{POSITION_NAMES_AR.get(r.pos_group, r.pos_group)} — {eur(r.market_value_in_eur)}"
                          for _, r in results.iterrows()}
                s1, s2 = st.columns([4, 1])
                choice = s1.selectbox("اختر اللاعب", list(labels), format_func=labels.get, label_visibility="collapsed")
                if s2.button("حلّل اللاعب", type="primary", use_container_width=True):
                    st.session_state.update(pid=choice, club=target_club.strip() or None, coach=coach.strip() or None,
                                            notes=notes.strip() or None, langs=langs or None, auto_ai=True, page=PAGES[0])
                    st.rerun()
    if not pid:
        st.markdown("<div class='ks-panel ks-empty'><div class='t'>ابحث عن لاعب للبدء</div><div class='ks-muted'>"
                    "ملخص موضوعي · أبرز الإيجابيات والسلبيات · الفرق التي تناسبه · توافقه مع المدرب · مسيرته وثباته · "
                    "قيمته وراتبه · بدلاء بنفس الأسلوب · تقرير PDF</div></div>", unsafe_allow_html=True)
        return
    player_view(pid)


def player_view(pid: int) -> None:
    club, coach_in, notes_in = st.session_state.get("club"), st.session_state.get("coach"), st.session_state.get("notes")
    base = get_base(pid)
    p = base["player"]
    cached = ai.load_cached(pid, club, coach_in, notes_in)
    run_now = ai_on and not cached and st.session_state.pop("auto_ai", False)
    if ai_on and not run_now:
        b1, b2 = st.columns([5, 1])
        if cached:
            b1.caption(f"آخر تحليل: {cached.get('_saved_at', '')}" + (f" · النادي: {club}" if club else "")
                       + (f" · المدرب: {coach_in}" if coach_in else "") + (" · مع ملاحظاتك" if notes_in else ""))
            run_now = b2.button("تحديث التحليل", use_container_width=True)
        else:
            run_now = b2.button("تشغيل التحليل", type="primary", use_container_width=True)
    if run_now:
        with st.status("جارٍ التحليل الذكي...", expanded=True) as status:
            try:
                settings.count_ai_use()
                cached = profile.run_ai(base, club, coach_in, progress=status.write, notes=notes_in,
                                        langs=st.session_state.get("langs"))
                status.update(label="اكتمل التحليل", state="complete", expanded=False)
                st.rerun()
            except ai.AIError as e:
                status.update(label="تعذّر إكمال التحليل الذكي", state="error")
                st.error(str(e))

    syn = (cached or {}).get("synthesis")
    comb = profile.combine(base, syn)
    v, cons, trend = base["valuation"], base["consistency"], base["trend"]
    league = LEAGUE_NAMES_AR.get(base["league_code"], base["league_code"])
    cf = (syn or {}).get("coach_fit")
    scores = {t["key"]: t["score"] for t in comb["traits"]}
    sm = (syn or {}).get("summary") or {}

    # ---------------------------------------------------------- hero
    h1, h2 = st.columns([1, 2.4])
    h1.markdown(kashaf_card(p, base["pos_group"], comb, v["base_value"], cons, base.get("team_color"),
                            base.get("photo"), base.get("team_logo")), unsafe_allow_html=True)
    contract = "—" if pd.isna(p["contract_expiration_date"]) else f"{p['contract_expiration_date']:%Y-%m}"
    chips = [POSITION_NAMES_AR.get(base["pos_group"]), f"{p['age']:.0f} سنة", p["country_of_citizenship"],
             f"القدم: {p['foot'] or '—'}", f"الطول: {p['height_in_cm'] or '—'} سم", f"العقد حتى {contract}"]
    if sm.get("confidence"):
        chips.append(f"ثقة التحليل: {sm['confidence']}")
    if base.get("injury"):
        chips.append(f"🩹 {base['injury']['injury']} · {base['injury']['expected_return']}")
    form = base["fotmob_matches"]
    kpis = "".join([
        kpi("القيمة العادلة", eur(v["base_value"]), "دمج 3 تقديرات"),
        kpi("سعر الشراء العادل", f"{eur(v['fee_low'])} - {eur(v['fee_high'])}", v["contract_note"][:30]),
        kpi("الراتب السنوي العادل", f"{eur(v['salary_low'])} - {eur(v['salary_high'])}", "إجمالي"),
        kpi("ثبات المستوى", f"{cons['score'] if cons['score'] is not None else '—'}/100", cons["label"]),
        kpi("آخر 10 مباريات", "—" if form.empty else f"{form.tail(10)['rating'].mean():.2f}",
            f"الاتجاه: {trend.get('performance', '—')}"),
    ])
    club_logo = (f"<img src='{esc(base['team_logo'])}' style='width:22px;height:22px;object-fit:contain;"
                 f"vertical-align:middle;margin-left:8px;background:#fff;border-radius:50%;padding:2px'>") if base.get("team_logo") else ""
    headline = sm.get("headline") or ("شغّل التحليل الذكي للحصول على الملخص الكامل." if ai_on else "اربط DeepSeek لإكمال التحليل.")
    hero_bg = (f"background:radial-gradient(120% 140% at 0% 0%, {base.get('team_color') or '#1E3A8A'}66 0%, transparent 50%), "
               f"linear-gradient(120deg,#0B1222 0%,#13203A 100%)")
    h2.markdown(flat(f"""<div class='ks-hero' style="{hero_bg}">
      <div class='nm'>{esc(p['name'])}</div><div class='sub'>{club_logo}{esc(p['current_club_name'])} · {esc(league)}</div>
      <div class='arch'>{esc(sm.get('archetype'))}</div><div class='head'>{esc(headline)}</div>
      <div>{''.join(f"<span class='gl'>{esc(c)}</span>" for c in chips)}</div>
      <div class='ks-kpis'>{kpis}</div></div>"""), unsafe_allow_html=True)
    if syn and syn.get("verdict"):
        h2.markdown(f"<div class='ks-verdict'><b>الخلاصة:</b> {esc(syn['verdict'])}</div>", unsafe_allow_html=True)

    st.write("")
    page = st.radio("الصفحات", PAGES, horizontal=True, key="page", label_visibility="collapsed")

    def alt_lines(r):
        return [f"القيمة <b>{eur(r.market_value_in_eur)}</b> · التقييم <b>{r.rating:.2f}</b>"]

    def alternatives_block(n_preview: int) -> None:
        alts = get_alternatives(pid, "all")
        if alts is None or alts.empty:
            st.caption("لا يوجد لاعبون بأسلوب مشابه ضمن الميزانية.")
            return
        player_grid(alts.head(n_preview), lambda r: f"تشابه {r.similarity:.0f}%", alt_lines, lambda r: r.why, key=f"alt{n_preview}")
        with st.expander("عرض كل البدلاء والتصفية حسب الدوري"):
            scope = st.radio("النطاق", ["all", "league", "arab"], horizontal=True, key=f"alt_scope{n_preview}",
                             format_func={"all": "كل الدوريات", "league": "نفس الدوري", "arab": "الدوريات العربية"}.get,
                             label_visibility="collapsed")
            more = get_alternatives(pid, scope)
            if more is None or more.empty:
                st.caption("لا يوجد بدلاء في هذا النطاق.")
            else:
                player_grid(more, lambda r: f"تشابه {r.similarity:.0f}%", alt_lines, lambda r: r.why, key=f"altall{n_preview}_{scope}")

    def coach_snapshot() -> None:
        vc = VERDICT_COLOR.get(cf["verdict"], "#CA8A04")
        reqs = "".join(
            f"<div class='ks-req'><div><b>{esc(r['requirement'])}</b></div>"
            f"<div><span class='ks-badge' style='background:{LEVEL_COLOR.get(r['player_level'])}1A;color:{LEVEL_COLOR.get(r['player_level'])}'>"
            f"{esc(r['player_level'])}</span></div><div class='ks-muted'>{esc(r['note'])}</div></div>" for r in cf["requirements"][:5])
        c1, c2 = st.columns([1, 3.2])
        c1.markdown(f"<div class='ks-panel' style='text-align:center'>{ring(cf['fit_score'], '%', color=vc, pct=cf['fit_score'] or 0)}"
                    f"<div style='margin-top:10px'><span class='ks-badge' style='background:{vc};color:#fff'>{esc(cf['verdict'])}</span></div>"
                    f"<div class='ks-muted' style='margin-top:8px'>{esc(cf['coach_name'])}<br>{esc(cf['club_name'])} · {esc(cf['formation'])}</div></div>",
                    unsafe_allow_html=True)
        c2.markdown(f"<div class='ks-panel'><div class='ks-muted' style='margin-bottom:6px'>الدور المتوقع: {esc(cf['role_for_player'])}</div>"
                    f"{reqs}</div>", unsafe_allow_html=True)

    if page == PAGES[0]:
        pros, cons_ = (syn or {}).get("pros") or [], (syn or {}).get("cons") or []
        if not pros and not cons_:
            pros = [{"title": t["label"], "detail": f"أفضل من {t['data_pct']}% من لاعبي مركزه", "trait": t["key"]}
                    for t in comb["traits"] if t["verdict"] == "قوة"][:5]
            cons_ = [{"title": t["label"], "detail": f"أفضل من {t['data_pct']}% فقط من لاعبي مركزه", "trait": t["key"]}
                     for t in sorted(comb["traits"], key=lambda t: t["score"]) if t["score"] <= 45][:4]
        c1, c2 = st.columns(2)
        c1.markdown(section("أبرز الإيجابيات") + "".join(
            pc_item(x["title"], x["detail"], report.consistent_score(scores, x["trait"], True), True) for x in pros), unsafe_allow_html=True)
        c2.markdown(section("أبرز السلبيات") + "".join(
            pc_item(x["title"], x["detail"], report.consistent_score(scores, x["trait"], False), False) for x in cons_), unsafe_allow_html=True)
        if syn and syn.get("suitable_teams"):
            c1, c2 = st.columns(2)
            c1.markdown(section("نوعية الفرق التي تناسبه") + "".join(
                f"<div class='ks-team'><b>{esc(t['profile'])}</b><span>{esc(t['why'])}</span></div>" for t in syn["suitable_teams"]),
                unsafe_allow_html=True)
            c2.markdown(section("نوعية الفرق التي لا تناسبه") + "".join(
                f"<div class='ks-team'><b>{esc(t['profile'])}</b><span>{esc(t['why'])}</span></div>" for t in syn.get("unsuitable_teams") or []),
                unsafe_allow_html=True)
        if cf:
            # Summary keeps the two parts that matter most: how the coach plays, and the fit analysis.
            vc = VERDICT_COLOR.get(cf["verdict"], "#CA8A04")
            methods = "".join(f"<li><b>{esc(m['principle'])}:</b> {esc(m['detail'])}</li>" for m in cf["coach_methods"])
            st.markdown(section(f"أسلوب {cf['coach_name']} ومدى توافقه", f"{cf['club_name']} · {cf['formation']}"),
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.markdown(f"<div class='ks-panel'><div class='ks-h' style='font-family:Tajawal;font-weight:800;margin-bottom:8px'>أسلوب المدرب</div>"
                        f"<ul class='ks-list'>{methods}</ul></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='ks-panel'><div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
                        f"<span style='font-family:Tajawal;font-weight:800'>التحليل</span>"
                        f"<span class='ks-badge' style='background:{vc};color:#fff'>{esc(cf['verdict'])}</span></div>"
                        f"<p style='margin:0'>{esc(cf['fit_analysis'])}</p></div>", unsafe_allow_html=True)
            st.button("تفاصيل المدرب والنادي ←", on_click=go_to, args=(PAGES[2],))
        st.markdown(section("بدلاء بنفس أسلوب اللعب", "تشابه في الأرقام (صناعة، تمرير، مراوغة، دفاع...) · قيمة أفضل"),
                    unsafe_allow_html=True)
        alternatives_block(3)
        if syn:
            risks = "".join(f"<li>{esc(r)}</li>" for r in syn.get("risks", [])[:4])
            gaps = "".join(f"<li>{esc(g)}</li>" for g in (sm.get("data_gaps") or [])[:3])
            c1, c2 = st.columns(2)
            if risks:
                c1.markdown(section("المخاطر") + f"<div class='ks-panel'><ul class='ks-list'>{risks}</ul></div>", unsafe_allow_html=True)
            if gaps:
                c2.markdown(section("ما لم تحسمه البيانات") + f"<div class='ks-panel'><ul class='ks-list'>{gaps}</ul></div>",
                            unsafe_allow_html=True)
            st.button("تحميل التقرير PDF ←", type="primary", on_click=go_to, args=(PAGES[5],))

    elif page == PAGES[1]:
        if syn:
            c1, c2 = st.columns(2)
            c1.markdown(section("أسلوب اللعب") + f"<div class='ks-panel'><p>{esc(syn['playing_style'])}</p></div>", unsafe_allow_html=True)
            c2.markdown(section("الدور والمنظومة الأنسب") + f"<div class='ks-panel'><p>{esc(syn['position_and_system'])}</p></div>",
                        unsafe_allow_html=True)
        else:
            st.info("وصف أسلوب اللعب يُكتب بعد التحليل الذكي.")
        c1, c2 = st.columns([1, 2.2])
        c1.markdown(section("المراكز التي لعبها") + "<div class='ks-panel'>" + pitch(comb["positions"])
                    + "<div class='ks-muted' style='text-align:center;margin-top:8px'>الرقم = عدد المباريات · الذهبي = المركز الأساسي</div></div>",
                    unsafe_allow_html=True)
        c2.markdown(section("سمات اللعب", "50 = متوسط مركزه في الدوري") + "<div class='ks-grid'>"
                    + "".join(trait_card(t) for t in comb["traits"]) + "</div>", unsafe_allow_html=True)
        if base["peer_metrics"]:
            rows = ""
            for m in base["peer_metrics"]:
                pct = None if m["pct"] is None or pd.isna(m["pct"]) else float(m["pct"])
                rank = "—" if pct is None else f"أفضل من {pct:.0f}%"
                rows += bar_row(m["metric"], pct, right=f"{m['value']:.2f} لكل 90 · {rank}", cols="220px 1fr 190px")
            st.markdown(section("الأرقام مقارنة بلاعبي مركزه", base["rating_season"] or "") + f"<div class='ks-panel'>{rows}</div>",
                        unsafe_allow_html=True)

    elif page == PAGES[2]:
        if not (club or coach_in):
            st.info("اكتب اسم المدرب و/أو النادي المستهدف في «تحليل لاعب آخر» ثم اضغط «حلّل اللاعب»: يُحلَّل أسلوب المدرب "
                    "وهوية النادي وأهدافه، ويُقارن كل ما يطلبه المدرب بمستوى اللاعب.")
        elif not cf:
            st.info("يكتمل تحليل المدرب والنادي بعد التحليل الذكي.")
        else:
            st.markdown(section(cf["coach_name"], f"{cf['club_name']} · {cf['formation']}"), unsafe_allow_html=True)
            coach_snapshot()
            methods = "".join(f"<li><b>{esc(m['principle'])}:</b> {esc(m['detail'])}</li>" for m in cf["coach_methods"])
            st.markdown(section("أساليب المدرب") + f"<div class='ks-panel'><ul class='ks-list'>{methods}</ul></div>", unsafe_allow_html=True)
            st.markdown(section("التحليل") + f"<div class='ks-panel'><p>{esc(cf['fit_analysis'])}</p>"
                        + (f"<p><b>هوية النادي:</b> {esc(cf['club_identity'])}</p>" if cf.get("club_identity") else "")
                        + (f"<p><b>أهداف النادي:</b> {esc(cf['club_objectives'])}</p>" if cf.get("club_objectives") else "")
                        + "</div>", unsafe_allow_html=True)

    elif page == PAGES[3]:
        seasons = pd.DataFrame(base["seasons"])
        if not seasons.empty:
            st.markdown(section("المسيرة موسمًا بموسم", "التقييم ومنحنى القيمة السوقية"), unsafe_allow_html=True)
            with st.container(border=True):
                seasons["x"] = seasons["label"] + " · " + seasons["club"].str.slice(0, 14)
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                if seasons["rating"].notna().any():
                    fig.add_bar(x=seasons["x"], y=seasons["rating"], name="متوسط التقييم",
                                marker_color=[rcolor(40 + (r - 6) * 25) if pd.notna(r) else "#CBD5E1" for r in seasons["rating"]],
                                text=seasons["rating"].round(2), textposition="outside")
                    fig.update_yaxes(range=[min(5.8, seasons["rating"].min() - .2), max(8.6, seasons["rating"].max() + .2)],
                                     secondary_y=False, **GRID)
                if seasons["market_value_end"].notna().any():
                    fig.add_scatter(x=seasons["x"], y=seasons["market_value_end"] / 1e6, name="القيمة (مليون €)",
                                    mode="lines+markers", line=dict(color="#C99A2E", width=3), secondary_y=True)
                fig.update_yaxes(secondary_y=True, **GRID)
                fig.update_layout(**PLOT, height=360, legend=dict(orientation="h", y=1.12), bargap=.35)
                st.plotly_chart(fig, use_container_width=True)
                rows = ""
                for s in reversed(base["seasons"]):
                    tags = "".join(f"<span class='ks-tag'>{esc(t)}</span>" for t in s["tags"])
                    rating = "—" if s["rating"] is None else f"{s['rating']:.2f}"
                    rows += (f"<tr><td>{esc(s['label'])}{' (جارٍ)' if s.get('ongoing') else ''}</td><td>{esc(s['club'])}</td>"
                             f"<td>{s['games']}</td><td>{s['goals']}</td><td>{s['assists']}</td><td>{rating}</td>"
                             f"<td>{eur(s['market_value_end'])}</td><td>{tags}</td></tr>")
                st.markdown(f"<div style='overflow-x:auto'><table class='ks-t'><thead><tr><th>الموسم</th><th>النادي</th><th>مباريات</th>"
                            f"<th>أهداف</th><th>صناعة</th><th>التقييم</th><th>القيمة</th><th>مؤشرات</th></tr></thead>"
                            f"<tbody>{rows}</tbody></table></div>", unsafe_allow_html=True)
        if syn:
            tl = "".join(f"<div class='it'><span class='pd'>{esc(ph['period'])}</span>"
                         f"<span class='ph' style='background:{PHASE_COLOR.get(ph['phase'], '#0EA5E9')}1F;color:{PHASE_COLOR.get(ph['phase'], '#0EA5E9')}'>"
                         f"{esc(ph['phase'])}</span><div>{esc(ph['explanation'])}</div></div>" for ph in syn["career_phases"])
            st.markdown(section("مراحل المسيرة وأسبابها") + f"<div class='ks-panel'><div class='ks-tl'>{tl}</div></div>", unsafe_allow_html=True)
        st.markdown(section("ثبات المستوى"), unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.7])
        c1.markdown(f"<div class='ks-panel' style='text-align:center'>{ring(cons['score'], cons['label'])}</div>"
                    + "<div class='ks-panel'>" + "".join(bar_row(k, val) for k, val in cons["parts"].items())
                    + "".join(f"<div class='ks-muted'>• {esc(n)}</div>" for n in cons["notes"]) + "</div>", unsafe_allow_html=True)
        with c2:
            fmm = base["fotmob_matches"]
            if not fmm.empty:
                with st.container(border=True):
                    last = fmm.tail(40)
                    fig = go.Figure(go.Scatter(x=last["date"], y=last["rating"], mode="lines+markers",
                                               marker=dict(size=9, color=[rcolor(40 + (r - 6) * 25) for r in last["rating"]]),
                                               line=dict(color="#CBD5E1"), text=last["opponent"] + " · " + last["league"],
                                               hoverinfo="text+y"))
                    fig.add_hline(y=last["rating"].mean(), line_dash="dot", line_color="#C99A2E",
                                  annotation_text=f"المتوسط {last['rating'].mean():.2f}")
                    fig.update_layout(**PLOT, height=300, title=dict(text="تقييم آخر المباريات", x=1, xanchor="right"))
                    fig.update_xaxes(**GRID)
                    fig.update_yaxes(**GRID)
                    st.plotly_chart(fig, use_container_width=True)
            if syn:
                st.markdown(f"<div class='ks-panel'><p><b>هل هناك هبوط؟</b> {esc(syn['decline_analysis'])}</p>"
                            f"<p><b>قراءة الثبات:</b> {esc(syn['consistency_analysis'])}</p></div>", unsafe_allow_html=True)

    elif page == PAGES[4]:
        st.markdown(section("القيمة والراتب") + "<div class='ks-tiles'>" + "".join([
            tile("سعر الشراء العادل", f"{eur(v['fee_low'])} - {eur(v['fee_high'])}", v["contract_note"]),
            tile("الراتب السنوي العادل", f"{eur(v['salary_low'])} - {eur(v['salary_high'])}",
                 f"حسب رواتب {LEAGUE_NAMES_AR.get(v['salary_league'], v['salary_league'])}"),
            tile("نموذجنا (الأداء والعمر والعقد)", eur(v["model_value"]),
                 "" if v.get("minutes_covered", True) else "وزن مخفّض لنقص بيانات الدقائق"),
            tile("Transfermarkt", eur(v["tm_value"])), tile("SciSports (الأحدث)", eur(v["scisports_value"])),
            tile("القيمة العادلة", eur(v["base_value"]), "متوسط مرجّح"),
        ]) + "</div>", unsafe_allow_html=True)
        if syn:
            st.markdown(f"<div class='ks-panel'><p>{esc(syn['valuation_view'])}</p>"
                        f"<div class='ks-muted'>الراتب المُعلن: {esc(syn['reported_salary'])}</div></div>", unsafe_allow_html=True)
            st.markdown(section("القيمة التسويقية والفنية"), unsafe_allow_html=True)
            c1, c2 = st.columns([1, 3.2])
            c1.markdown(f"<div class='ks-panel' style='text-align:center'>{ring(syn['marketing']['score'], 'من 100', pct=syn['marketing']['score'])}</div>",
                        unsafe_allow_html=True)
            c2.markdown(f"<div class='ks-panel'><p><b>{esc(syn['marketing']['social_reach'])}</b></p><p>{esc(syn['marketing']['analysis'])}</p>"
                        f"<p>{esc(syn['technical_value'])}</p></div>", unsafe_allow_html=True)
        st.markdown(section("بدلاء بنفس أسلوب اللعب"), unsafe_allow_html=True)
        alternatives_block(6)

    elif page == PAGES[5]:
        if not syn:
            st.info("التقرير يُجهَّز بعد التحليل الذكي.")
        else:
            st.markdown(section("تحميل التقرير", "تصميم جاهز للطباعة والمشاركة"), unsafe_allow_html=True)
            saved, fname = cached.get("_saved_at"), p["name"].replace(" ", "_")
            c1, c2 = st.columns(2)
            with c1, st.container(border=True):
                st.markdown("**الملخص**  \nالبطاقة والمؤشرات، الإيجابيات والسلبيات، الفرق المناسبة، التوافق مع المدرب، البدلاء والمخاطر.")
                try:
                    st.download_button("تحميل الملخص PDF", get_pdf(pid, club, coach_in, notes_in, saved, False),
                                       file_name=f"كشاف_ملخص_{fname}.pdf", mime="application/pdf", type="primary", use_container_width=True)
                except report.ReportError as e:
                    st.error(str(e))
            with c2, st.container(border=True):
                st.markdown("**التقرير الكامل**  \nكل ما في الملخص + أسلوب اللعب، سمات اللعب، الأرقام، المدرب والنادي، المسيرة، القيمة والتسويق.")
                try:
                    st.download_button("تحميل التقرير الكامل PDF", get_pdf(pid, club, coach_in, notes_in, saved, True),
                                       file_name=f"كشاف_تقرير_كامل_{fname}.pdf", mime="application/pdf", use_container_width=True)
                except report.ReportError as e:
                    st.error(str(e))
            ss = syn.get("sources_summary") or {}
            labels = {"analysts": "ماذا يقول المحللون", "fans": "انطباع الجمهور", "technical": "التقارير الفنية",
                      "matches": "أداؤه في المباريات", "stats": "ماذا تقول الأرقام"}
            if ss:
                st.markdown(section("ماذا تقول المصادر") + "<div class='ks-panel'>" + "".join(
                    f"<p><b>{labels[k]}:</b> {esc(ss[k])}</p>" for k in labels if ss.get(k)) + "</div>", unsafe_allow_html=True)

    elif page == PAGES[6]:
        st.markdown(section("بيانات المباريات والقيم") + "<div class='ks-panel'>"
                    f"<a href='{esc(p['url'])}' target='_blank'>Transfermarkt</a>"
                    + (f" · <a href='https://www.fotmob.com/players/{base['fotmob_id']}' target='_blank'>FotMob</a>" if base["fotmob_id"] else "")
                    + (f" · <a href='https://understat.com/player/{base['understat_id']}' target='_blank'>Understat</a>" if base["understat_id"] else "")
                    + "</div>", unsafe_allow_html=True)
        srcs = (cached or {}).get("sources") or []
        if not srcs:
            st.caption("تظهر مصادر التحليل الذكي هنا بعد تشغيله.")
        by_cat: dict[str, list] = {}
        for s in srcs:
            by_cat.setdefault(s["category"], []).append(s)
        for cat, label in web.CATEGORIES.items():
            items = by_cat.get(cat)
            if items:
                links = "".join(f"<li>{'📄 ' if s['read'] else ''}<a href='{esc(s['url'])}' target='_blank'>{esc(s['title'] or s['url'])}</a></li>"
                                for s in items)
                st.markdown(section(label, f"{len(items)} مصدر") + f"<div class='ks-panel'><ul class='ks-list'>{links}</ul></div>",
                            unsafe_allow_html=True)
        if syn:
            u = syn.get("_usage", {})
            st.caption(f"📄 = صفحة قُرئت بالكامل · استهلاك DeepSeek: {u.get('prompt_tokens', 0):,} توكن إدخال، "
                       f"{u.get('completion_tokens', 0):,} توكن إخراج")


if mode == MODES[1]:
    spec_mode()
else:
    analysis_mode()
