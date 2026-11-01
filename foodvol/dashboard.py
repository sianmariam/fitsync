"""Integrated dashboard: food (from the depth pipeline) + targets (BMR/TDEE) + workouts + weight, in one view."""
from __future__ import annotations

import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

from . import planner

TEAL, MINT, GOLD, CORAL, INK, MUTED = "#1F6B5E", "#3CB89A", "#E6A817", "#E4572E", "#14313A", "#6B8A90"

CSS = f"""
<style>
.kpi {{background:#fff;border:1px solid #E3ECEB;border-radius:14px;padding:14px 16px;height:100%;}}
.kpi .lab {{font-size:.80rem;color:{MUTED};text-transform:uppercase;letter-spacing:.04em;}}
.kpi .val {{font-size:1.75rem;font-weight:700;color:{INK};line-height:1.2;margin-top:2px;}}
.kpi .sub {{font-size:.82rem;color:{MUTED};}}
.kpi.good .val {{color:{TEAL};}} .kpi.warn .val {{color:{CORAL};}}
.hero {{background:linear-gradient(120deg,#14313A 0%,#1F6B5E 100%);color:#fff;border-radius:16px;padding:18px 22px;
        margin-bottom:14px;}}
.hero h2 {{margin:0;color:#fff;font-size:1.5rem;}} .hero p {{margin:.2rem 0 0;color:#CDE7E1;}}
.pill {{display:inline-block;background:rgba(255,255,255,.16);border-radius:999px;padding:2px 10px;margin-right:6px;
        font-size:.8rem;color:#fff;}}
.insight {{border-radius:12px;padding:10px 14px;margin-bottom:8px;background:#F3F8F7;border:1px solid #E3ECEB;}}
.insight b {{color:{INK};}}
.meal {{display:flex;gap:12px;align-items:center;border:1px solid #E3ECEB;border-radius:12px;padding:8px;margin-bottom:8px;
        background:#fff;}}
.meal img {{width:64px;height:64px;object-fit:cover;border-radius:10px;}}
.meal .ph {{width:64px;height:64px;border-radius:10px;background:#E7F2EF;display:flex;align-items:center;
            justify-content:center;font-size:1.6rem;}}
.meal .t {{font-weight:600;color:{INK};}} .meal .s {{font-size:.82rem;color:{MUTED};}}
.meal .k {{margin-left:auto;font-weight:700;color:{TEAL};white-space:nowrap;}}
</style>"""

EMOJI = {"Breakfast": "🌅", "Lunch": "🍛", "Snack": "🍎", "Dinner": "🌙"}


def kpi(col, label, value, sub="", tone=""):
    col.markdown(f'<div class="kpi {tone}"><div class="lab">{label}</div><div class="val">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def day_totals(d, date):
    meals = [m for m in d["meals"] if m["date"] == date]
    wos = [w for w in d["workouts"] if w["date"] == date]
    return dict(kcal=sum(m["kcal"] for m in meals), protein=sum(m["protein"] for m in meals),
                carbs=sum(m["carbs"] for m in meals), fat=sum(m["fat"] for m in meals),
                burned=sum(w["kcal"] for w in wos), minutes=sum(w["minutes"] for w in wos), meals=meals, workouts=wos)


def streak(d):
    days = {m["date"] for m in d["meals"]}
    n, cur = 0, dt.date.today()
    if cur.isoformat() not in days:
        cur -= dt.timedelta(days=1)
    while cur.isoformat() in days:
        n += 1
        cur -= dt.timedelta(days=1)
    return n


def ring(eaten, target, burned):
    remaining = max(0.0, target + burned - eaten)
    over = max(0.0, eaten - target - burned)
    df = pd.DataFrame({"part": ["Eaten", "Remaining", "Over"], "kcal": [min(eaten, target + burned), remaining, over]})
    base = alt.Chart(df).encode(theta=alt.Theta("kcal:Q", stack=True),
                                color=alt.Color("part:N", scale=alt.Scale(domain=["Eaten", "Remaining", "Over"],
                                                                          range=[TEAL, "#E3ECEB", CORAL]), legend=None),
                                tooltip=["part", alt.Tooltip("kcal:Q", format=".0f")])
    arc = base.mark_arc(innerRadius=78, outerRadius=105, cornerRadius=4)
    big = f"{remaining:.0f}" if not over else f"+{over:.0f}"
    txt = alt.Chart(pd.DataFrame({"t": [big]})).mark_text(fontSize=30, fontWeight="bold", color=INK, dy=-8).encode(
        text="t:N")
    sub = alt.Chart(pd.DataFrame({"t": ["kcal left" if not over else "kcal over"]})).mark_text(
        fontSize=13, color=MUTED, dy=18).encode(text="t:N")
    return (arc + txt + sub).properties(width=240, height=240)


def weekly_chart(d, target):
    rows = []
    for back in range(6, -1, -1):
        day = dt.date.today() - dt.timedelta(days=back)
        t = day_totals(d, day.isoformat())
        lab = day.strftime("%a %d")
        rows += [dict(day=lab, order=6 - back, kind="Eaten", kcal=t["kcal"]),
                 dict(day=lab, order=6 - back, kind="Burned (exercise)", kcal=t["burned"])]
    df = pd.DataFrame(rows)
    sort = list(dict.fromkeys(df.sort_values("order").day))
    bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("day:N", sort=sort, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("kind:N", sort=["Eaten", "Burned (exercise)"]),
        y=alt.Y("kcal:Q", title="kcal"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Eaten", "Burned (exercise)"], range=[TEAL, GOLD]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["day", "kind", alt.Tooltip("kcal:Q", format=".0f")])
    rule = alt.Chart(pd.DataFrame({"y": [target]})).mark_rule(strokeDash=[6, 4], color=CORAL, size=2).encode(y="y:Q")
    lab = alt.Chart(pd.DataFrame({"y": [target], "t": [f"target {target:.0f}"]})).mark_text(
        align="left", dx=4, dy=-7, color=CORAL, fontSize=11).encode(y="y:Q", text="t:N", x=alt.value(0))
    return (bars + rule + lab).properties(height=260)


def weight_chart(d):
    if len(d["weights"]) < 2:
        return None
    df = pd.DataFrame(d["weights"])[-30:]
    df["date"] = pd.to_datetime(df["date"])
    lo, hi = df.kg.min() - 0.5, df.kg.max() + 0.5
    line = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(filled=True, size=50, color=TEAL), color=TEAL, strokeWidth=2.5).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b")),
        y=alt.Y("kg:Q", scale=alt.Scale(domain=[lo, hi]), title="kg"),
        tooltip=[alt.Tooltip("date:T", format="%d %b"), "kg"])
    return line.properties(height=260)


def insights(d, p, tgt, macros, t, session, done_names):
    out = []
    remaining = tgt["kcal"] + t["burned"] - t["kcal"]
    hour = dt.datetime.now().hour
    pgap = macros["protein"] - t["protein"]
    if pgap > 15:
        out.append(f"🥚 <b>{pgap:.0f} g protein to go.</b> 150 g paneer ≈ 27 g · 2 eggs ≈ 13 g · 1 bowl dal ≈ 13 g.")
    if remaining < 0:
        mins = planner.burn_minutes(-remaining, p["weight"])
        out.append(f"🔥 <b>{-remaining:.0f} kcal over target.</b> Balance it with ~{mins['Brisk walking']:.0f} min brisk "
                   f"walking or ~{mins['Jogging']:.0f} min jogging.")
    elif hour >= 17 and remaining > 500:
        out.append(f"🍽️ <b>{remaining:.0f} kcal still available</b> — don't skip dinner; under-eating slows progress.")
    if session:
        left = [e for e in session["exercises"] if e["Exercise"] not in done_names]
        if left:
            est = sum(planner.exercise_minutes_kcal(e, p["weight"])[1] for e in left)
            out.append(f"🏋️ <b>{len(left)} exercise{'s' if len(left) != 1 else ''} left today</b> ({session['focus']}) — about {est:.0f} kcal more to burn.")
        else:
            out.append("✅ <b>Today's workout is complete.</b> Nice work — prioritise protein and sleep for recovery.")
    else:
        out.append("🧘 <b>Rest day.</b> A 20–30 min easy walk helps recovery without affecting your plan.")
    # weekly projection
    days = [day_totals(d, (dt.date.today() - dt.timedelta(days=b)).isoformat()) for b in range(1, 8)]
    logged = [x for x in days if x["kcal"] > 0]
    if len(logged) >= 3:
        avg_net = sum(x["kcal"] - x["burned"] for x in logged) / len(logged)
        delta = (avg_net - tgt["tdee"]) * 7 / 7700
        trend = "gain" if delta > 0 else "loss"
        aligned = (p["goal"] == "Lose weight" and delta < 0) or (p["goal"] == "Gain muscle" and delta > 0) or \
                  (p["goal"] == "Maintain" and abs(delta) < 0.15)
        out.append(f"📈 <b>Last {len(logged)} days:</b> average net intake {avg_net:.0f} kcal vs TDEE {tgt['tdee']:.0f} → "
                   f"projected {abs(delta):.2f} kg/week {trend} — "
                   + ("on track for your goal." if aligned else "adjust portions to match your goal."))
    return out


def render(d, p):
    st.markdown(CSS, unsafe_allow_html=True)
    tgt = planner.targets(p["age"], p["sex"], p["height"], p["weight"], p["activity"], p["goal"])
    macros = planner.macro_targets(tgt["kcal"], tgt["protein_g"])
    t = day_totals(d, dt.date.today().isoformat())
    b, bcat = planner.bmi(p["weight"], p["height"])
    session, day_name = planner.todays_session(p["goal"], p["days"], p["level"], p["equip"])
    done_names = {w["name"] for w in t["workouts"]}
    hour = dt.datetime.now().hour
    greet = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"

    st.markdown(f'<div class="hero"><h2>{greet}, {p.get("name") or "there"} 👋</h2>'
                f'<p>{dt.date.today().strftime("%A, %d %B %Y")}</p><div style="margin-top:8px">'
                f'<span class="pill">🎯 {p["goal"]}</span><span class="pill">🔥 {streak(d)}-day logging streak</span>'
                f'<span class="pill">🏋️ {session["focus"] if session else "Rest day"}</span></div></div>',
                unsafe_allow_html=True)

    remaining = tgt["kcal"] + t["burned"] - t["kcal"]
    c = st.columns(6)
    kpi(c[0], "Eaten", f"{t['kcal']:.0f}", f"of {tgt['kcal']:.0f} kcal target")
    kpi(c[1], "Burned", f"{t['burned']:.0f}", f"{t['minutes']:.0f} min exercise", "good" if t["burned"] else "")
    kpi(c[2], "Remaining", f"{remaining:.0f}", "kcal left incl. exercise", "warn" if remaining < 0 else "good")
    kpi(c[3], "Protein", f"{t['protein']:.0f} g", f"of {macros['protein']:.0f} g",
        "good" if t["protein"] >= macros["protein"] else "")
    kpi(c[4], "BMI", f"{b:.1f}", bcat + " (Asian cut-offs)", "good" if bcat == "Normal" else "warn")
    kpi(c[5], "TDEE", f"{tgt['tdee']:.0f}", f"BMR {tgt['bmr']:.0f} × activity")
    st.write("")

    left, mid, right = st.columns([1.05, 1.1, 1.25])
    with left:
        st.markdown("##### Today's calories")
        st.altair_chart(ring(t["kcal"], tgt["kcal"], t["burned"]), use_container_width=True)
    with mid:
        st.markdown("##### Macros")
        for k, lab, col in (("protein", "Protein", TEAL), ("carbs", "Carbs", GOLD), ("fat", "Fat", CORAL)):
            v, goal = t[k], macros[k]
            st.progress(min(1.0, v / goal if goal else 0), text=f"{lab}: {v:.0f} / {goal:.0f} g")
        st.markdown("##### Today's workout")
        if session:
            n_done = sum(e["Exercise"] in done_names for e in session["exercises"])
            st.progress(n_done / len(session["exercises"]),
                        text=f"{day_name} · {session['focus']}: {n_done}/{len(session['exercises'])} done")
            st.caption("Tick exercises off in the Workout tab — burned calories flow back here.")
        else:
            st.info(f"{day_name}: rest day in your {p['days']}-day plan.")
    with right:
        st.markdown("##### Smart insights")
        for s in insights(d, p, tgt, macros, t, session, done_names):
            st.markdown(f'<div class="insight">{s}</div>', unsafe_allow_html=True)

    st.write("")
    w1, w2 = st.columns([1.6, 1])
    with w1:
        st.markdown("##### Last 7 days — eaten vs burned")
        st.altair_chart(weekly_chart(d, tgt["kcal"]), use_container_width=True)
    with w2:
        st.markdown("##### Weight trend")
        wc = weight_chart(d)
        if wc is not None:
            st.altair_chart(wc, use_container_width=True)
        else:
            st.info("Log your weight in Profile on a couple of days to see the trend.")

    st.markdown("##### Today's meals")
    if not t["meals"]:
        st.info("No meals yet today — analyse a food photo and add it to a meal.")
    else:
        cols = st.columns(2)
        for i, m in enumerate(sorted(t["meals"], key=lambda m: m["time"])):
            img = (f'<img src="data:image/jpeg;base64,{m["thumb"]}">' if m.get("thumb")
                   else f'<div class="ph">{EMOJI.get(m["meal"], "🍽️")}</div>')
            names = ", ".join(f'{x["food"].replace("_", " ")} ({x["grams"]} g)' for x in m["items"])
            cols[i % 2].markdown(f'<div class="meal">{img}<div><div class="t">{m["meal"]} · {m["time"]}</div>'
                                 f'<div class="s">{names}</div><div class="s">P {m["protein"]:.0f} g · C {m["carbs"]:.0f} g'
                                 f' · F {m["fat"]:.0f} g</div></div><div class="k">{m["kcal"]:.0f} kcal</div></div>',
                                 unsafe_allow_html=True)
