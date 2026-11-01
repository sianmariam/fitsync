"""FitSync — depth-aware calorie estimation + workout planner + integrated dashboard.

Run:  streamlit run app.py
"""
import os
from dataclasses import asdict

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from foodvol.core import FoodVolumePipeline, Det
import datetime as dt

from foodvol import gemini, planner, store, dashboard, auth

st.set_page_config(page_title="FitSync", page_icon="🥗", layout="wide")

COINS = {"₹1 coin (20 mm)": 2.0, "₹2 coin (23 mm)": 2.3, "₹5 coin (25 mm)": 2.5, "₹10 coin (27 mm)": 2.7,
         "¥1 coin — ECUSTFD dataset (25 mm)": 2.5, "Custom": None}


def ensure_depth_model():
    """On cloud hosting the 95 MB depth model isn't in the repository: download it once from the URL
    given in the DA2_URL secret / environment variable (e.g. the GitHub Release asset link)."""
    p = store.ROOT / "models" / "da2_vits.onnx"
    if p.exists():
        return
    url = os.environ.get("DA2_URL")
    try:
        url = st.secrets.get("DA2_URL", url)
    except Exception:  # no secrets file (running locally)
        pass
    if not url:
        st.error("Depth model models/da2_vits.onnx is missing. Put it in the models folder, "
                 "or set DA2_URL to its download link.")
        st.stop()
    import urllib.request
    with st.spinner("Downloading the depth model (95 MB, first start only)…"):
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(p)


@st.cache_resource(show_spinner="Loading deep learning models…")
def load_pipeline():
    ensure_depth_model()
    return FoodVolumePipeline()


def colorize(x, mask=None):
    v = x.copy()
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
    v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    c = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def overlay(bgr, res):
    img = bgr.copy()
    palette = [(46, 204, 113), (52, 152, 219), (231, 76, 60), (241, 196, 15), (155, 89, 182)]
    for i, (it, m) in enumerate(zip(res["items"], res["masks"])):
        col = palette[i % len(palette)]
        layer = img.copy()
        layer[m] = col
        img = cv2.addWeighted(layer, 0.45, img, 0.55, 0)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, col, 2)
        x1, y1 = int(it.box[0]), int(it.box[1])
        label = f"{it.name} {it.volume_cm3:.0f}cm3 {it.kcal:.0f}kcal"
        cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    if res.get("coin"):
        b = [int(v) for v in res["coin"].box]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 215, 255), 2)
        cv2.putText(img, "reference coin", (b[0], b[3] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- login
auth.ensure_demo()
if "user" not in st.session_state:
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.title("🥗 FitSync")
        st.caption("Depth-aware food calorie estimation and workout planner")
        t_in, t_up = st.tabs(["Log in", "Sign up"])
        with t_in:
            with st.form("login"):
                u = st.text_input("Username")
                pw = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", type="primary", use_container_width=True):
                    if auth.log_in(u, pw):
                        st.session_state.user = u.strip().lower()
                        st.rerun()
                    else:
                        st.error("Wrong username or password.")
            st.caption(f"Demo account: **{auth.DEMO_USER}** / **{auth.DEMO_PASS}**")
        with t_up:
            with st.form("signup"):
                nm = st.text_input("Your name")
                u2 = st.text_input("Choose a username")
                p1 = st.text_input("Password (min 6 characters)", type="password")
                p2 = st.text_input("Confirm password", type="password")
                if st.form_submit_button("Create account", type="primary", use_container_width=True):
                    if p1 != p2:
                        st.error("Passwords don't match.")
                    else:
                        err = auth.sign_up(u2, p1, nm)
                        if err:
                            st.error(err)
                        else:
                            st.session_state.user = u2.strip().lower()
                            st.session_state.new_user = True
                            st.rerun()
    st.stop()

USER = st.session_state.user

# ---------------------------------------------------------------- sidebar + data
data = store.load(USER, auth.display_name(USER))
P = data["profile"]

# ---------------------------------------------------------------- first-time profile setup
if not P.get("setup_done"):
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.title("👋 Welcome to FitSync")
        st.caption(f"Logged in as @{USER}. Tell us about yourself so we can calculate your daily calorie target, "
                   "macros and workout plan. You can change these later in **Profile & targets**.")
        levels, equips = ["Beginner", "Intermediate", "Advanced"], ["None (home)", "Dumbbells", "Full gym"]
        with st.form("setup"):
            st.subheader("Your details")
            nm = st.text_input("Full name", P.get("name") if P.get("name") not in (None, "", USER) else auth.display_name(USER))
            a1, a2 = st.columns(2)
            age = a1.number_input("Age (years)", 10, 100, int(P["age"]))
            sex = a2.radio("Sex", ["Female", "Male"], index=["Female", "Male"].index(P["sex"]), horizontal=True)
            ht = a1.number_input("Height (cm)", 100, 230, int(P["height"]))
            wt = a2.number_input("Weight (kg)", 30.0, 250.0, float(P["weight"]), 0.1)
            st.subheader("Your goal")
            g1, g2 = st.columns(2)
            act = g1.selectbox("Activity level", list(planner.ACTIVITY), index=list(planner.ACTIVITY).index(P["activity"]))
            goal = g2.selectbox("Goal", list(planner.GOALS), index=list(planner.GOALS).index(P["goal"]))
            st.subheader("Your workouts")
            w1, w2, w3 = st.columns(3)
            days = w1.slider("Days per week", 2, 6, int(P["days"]))
            lvl = w2.selectbox("Level", levels, index=levels.index(P["level"]))
            eq = w3.selectbox("Equipment", equips, index=equips.index(P["equip"]))
            if st.form_submit_button("Save and continue →", type="primary", use_container_width=True):
                P.update(name=nm.strip() or USER, age=int(age), sex=sex, height=int(ht), weight=float(wt), activity=act,
                         goal=goal, days=int(days), level=lvl, equip=eq, setup_done=True)
                store.log_weight(data, float(wt))      # also saves
                t = planner.targets(P["age"], P["sex"], P["height"], P["weight"], P["activity"], P["goal"])
                st.session_state.welcome = (f"Profile saved! Your BMR is **{t['bmr']:.0f} kcal**, TDEE **{t['tdee']:.0f} kcal**, "
                                            f"and your daily target is **{t['kcal']:.0f} kcal** with **{t['protein_g']:.0f} g** protein.")
                st.rerun()
        if st.button("Log out"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    st.stop()

st.sidebar.title("🥗 FitSync")
st.sidebar.caption("Depth-aware food volume & calorie estimation")
sb1, sb2 = st.sidebar.columns([3, 2])
sb1.markdown(f"👤 **{P.get('name') or auth.display_name(USER)}**  \n<small>@{USER}</small>", unsafe_allow_html=True)
if sb2.button("Log out", use_container_width=True):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()
st.session_state.pop("new_user", None)
if "welcome" in st.session_state:
    st.success(st.session_state.pop("welcome"))
st.sidebar.divider()
coin_choice = st.sidebar.selectbox("Reference coin in photo", list(COINS), index=3)
coin_cm = COINS[coin_choice] or st.sidebar.number_input("Coin diameter (cm)", 1.0, 4.0, 2.5, 0.05)
hfov = st.sidebar.slider("Camera horizontal field of view (°)", 50, 85, 66,
                         help="Typical phone main camera ≈ 65–70°. Used to convert coin size to camera distance.")
use_gemini = st.sidebar.checkbox("Use Gemini to name foods the detector doesn't know", value=False)
api_key = st.sidebar.text_input("Gemini API key", type="password", value=os.environ.get("GEMINI_API_KEY", "")) \
    if use_gemini else None
with st.sidebar.expander("Demo data"):
    st.caption("Fills the previous 6 days with sample meals, workouts and weights so the weekly charts have "
               "something to show. Sample entries are marked and can be removed.")
    if st.button("Load sample week"):
        t0 = planner.targets(P["age"], P["sex"], P["height"], P["weight"], P["activity"], P["goal"])
        store.seed_demo_week(data, t0["kcal"], P["weight"])
        st.rerun()
    if st.button("Remove sample data"):
        store.clear_demo(data)
        st.rerun()

tab_dash, tab_food, tab_workout, tab_profile, tab_hist = st.tabs(
    ["🏠 Dashboard", "🍽️ Analyse food", "🏋️ Workout", "👤 Profile & targets", "📅 History"])

# ---------------------------------------------------------------- dashboard
with tab_dash:
    dashboard.render(data, P)

# ---------------------------------------------------------------- food tab
with tab_food:
    st.subheader("Photograph your food from above with a coin beside the plate")
    c1, c2 = st.columns(2)
    up = c1.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    cam = c2.camera_input("…or take one") if c2.toggle("Use camera") else None
    src = up or cam
    if src is not None:
        bgr = cv2.imdecode(np.frombuffer(src.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if max(bgr.shape[:2]) > 1280:
            s = 1280 / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        pipe = load_pipeline()
        n_det = pipe.detector.n_models if pipe.detector else 0
        names = ["ECUSTFD: 19 foods + coin"]
        if (store.ROOT / "models" / "idli_dosa_yolov8n.pt").exists():
            names.append("Indian: idli, dosa, chutney")
        if (store.ROOT / "models" / "indian_food_yolo11n.pt").exists():
            names.append("Indian dishes: 32 classes (YOLO11n)")
        if (store.ROOT / "models" / "indian_food_yolov8.pt").exists():
            names.append("IndianFoodNet: 30 dishes")
        st.caption("Detectors loaded — " + " · ".join(names))
        extra, gem_items = None, []
        if use_gemini and api_key:
            with st.spinner("Gemini: identifying foods…"):
                try:
                    gem_items = gemini.analyze(bgr, api_key)
                    extra = gemini.to_dets(gem_items, bgr.shape)
                    for g in gem_items:
                        if pipe.db.lookup(g["name"]) is None and "kcal_100g" in g:
                            pipe.db.add(g["name"], g.get("density_g_cm3", 0.8), g["kcal_100g"],
                                        g.get("protein_100g", 0), g.get("carbs_100g", 0), g.get("fat_100g", 0))
                except Exception as e:  # noqa
                    st.warning(f"Gemini unavailable ({e}). Continuing with the on-device detector only.")
        with st.spinner("Detecting → segmenting → estimating depth → integrating volume…"):
            res = pipe.run(bgr, extra_dets=extra, coin_diam_cm=coin_cm, hfov_deg=hfov)
        if res["error"]:
            st.error(res["error"])
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            a, b, c = st.columns(3)
            a.image(overlay(bgr, res), caption="1–2 · Detection + segmentation", use_container_width=True)
            b.image(colorize(res["disp"]), caption="3 · Depth Anything V2 (relative depth)", use_container_width=True)
            c.image(colorize(res["hgt"]), caption="4–5 · Height above table (cm) inside food masks",
                    use_container_width=True)
            if res.get("approx"):
                st.warning("No coin found, so sizes are estimated assuming the phone was about 35 cm above the food. "
                           "Accuracy is lower; add a coin for a measured result.")
            if not res["items"]:
                st.warning("Coin found but no food detected. Try Gemini naming, or move closer.")
            else:
                cal = res["calib"]
                st.caption(f"Calibration: {cal.px_per_cm:.1f} px/cm · {'assumed' if res.get('approx') else 'estimated'} "
                           f"camera distance {cal.cam_dist_cm:.0f} cm")
                gmap = {g["name"]: g.get("gemini_grams") for g in gem_items}
                rows = []
                for it in res["items"]:
                    r = dict(Food=it.name.replace("_", " "), Confidence=round(it.conf, 2),
                             **{"Volume (cm³)": round(it.volume_cm3), "Mass (g)": round(it.mass_g),
                                "Calories (kcal)": round(it.kcal), "Protein (g)": round(it.protein_g, 1),
                                "Carbs (g)": round(it.carbs_g, 1), "Fat (g)": round(it.fat_g, 1)})
                    if gmap:
                        r["Gemini guess (g)"] = gmap.get(it.name)
                    rows.append(r)
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                tot = df[["Calories (kcal)", "Protein (g)", "Carbs (g)", "Fat (g)"]].sum()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total calories", f"{tot['Calories (kcal)']:.0f} kcal")
                m2.metric("Protein", f"{tot['Protein (g)']:.1f} g")
                m3.metric("Carbs", f"{tot['Carbs (g)']:.1f} g")
                m4.metric("Fat", f"{tot['Fat (g)']:.1f} g")
                with st.expander("How was this measured? (features fed to the ML correction model)"):
                    st.dataframe(pd.DataFrame([it.features for it in res["items"]]).round(2),
                                 use_container_width=True)
                mc1, mc2 = st.columns([1, 2])
                hour = dt.datetime.now().hour
                default_meal = 0 if hour < 11 else 1 if hour < 16 else 2 if hour < 19 else 3
                meal_type = mc1.selectbox("Meal", ["Breakfast", "Lunch", "Snack", "Dinner"], index=default_meal)
                if mc2.button("➕ Add to today's log", type="primary"):
                    items = [dict(food=it.name, grams=round(it.mass_g), kcal=round(it.kcal),
                                  protein=round(it.protein_g, 1), carbs=round(it.carbs_g, 1), fat=round(it.fat_g, 1))
                             for it in res["items"]]
                    store.add_meal(data, meal_type, items, bgr)
                    st.success(f"Added to {meal_type}. See the Dashboard.")

# ---------------------------------------------------------------- workout tab
with tab_workout:
    c1, c2, c3 = st.columns(3)
    P["days"] = c1.slider("Days per week", 2, 6, int(P["days"]))
    levels, equips = ["Beginner", "Intermediate", "Advanced"], ["None (home)", "Dumbbells", "Full gym"]
    P["level"] = c2.selectbox("Level", levels, index=levels.index(P["level"]))
    P["equip"] = c3.selectbox("Equipment", equips, index=equips.index(P["equip"]))
    store.save(data)
    session, day_name = planner.todays_session(P["goal"], P["days"], P["level"], P["equip"])
    st.markdown(f"#### Today ({day_name})" + (f" — {session['focus']}" if session else " — rest day"))
    done_today = {w["name"] for w in data["workouts"] if w["date"] == store.today()}
    if session:
        st.caption("Tick each exercise when you finish it. Estimated calories (MET × body weight × time) "
                   "go straight to the Dashboard.")
        for ex in session["exercises"]:
            mins, kc = planner.exercise_minutes_kcal(ex, P["weight"])
            cc = st.columns([3, 2, 1.2])
            checked = cc[0].checkbox(f"**{ex['Exercise']}** — {ex['Sets']} × {ex['Reps']} (rest {ex['Rest']})",
                                     value=ex["Exercise"] in done_today, key=f"ex_{ex['Exercise']}")
            cc[1].caption(f"≈ {mins:.0f} min · {kc:.0f} kcal")
            if checked and ex["Exercise"] not in done_today:
                store.add_workout(data, ex["Exercise"], round(mins), round(kc))
                st.rerun()
            if not checked and ex["Exercise"] in done_today:
                store.remove_workout(data, ex["Exercise"])
                st.rerun()
    with st.expander("➕ Log other activity"):
        a1, a2, a3 = st.columns([2, 1, 1])
        act = a1.selectbox("Activity", list(planner.METS))
        mins = a2.number_input("Minutes", 5, 240, 30, 5)
        kc = planner.METS[act] * 3.5 * P["weight"] / 200 * mins
        a3.metric("≈ kcal", f"{kc:.0f}")
        if st.button("Log activity"):
            store.add_workout(data, f"{act} ({mins} min)", int(mins), round(kc))
            st.rerun()
    st.markdown("#### Full weekly plan")
    for i, d in enumerate(planner.workout_plan(P["goal"], P["days"], P["level"], P["equip"])):
        with st.expander(f"**{d['day']} — {d['focus']}**", expanded=False):
            st.table(pd.DataFrame(d["exercises"]))

# ---------------------------------------------------------------- profile tab
with tab_profile:
    c1, c2, c3 = st.columns(3)
    P["name"] = c1.text_input("Name", P.get("name", ""))
    P["age"] = c1.number_input("Age", 10, 100, int(P["age"]))
    P["sex"] = c1.radio("Sex", ["Female", "Male"], index=["Female", "Male"].index(P["sex"]), horizontal=True)
    P["height"] = c2.number_input("Height (cm)", 100, 230, int(P["height"]))
    P["weight"] = c2.number_input("Weight (kg)", 30.0, 250.0, float(P["weight"]), 0.1)
    P["activity"] = c3.selectbox("Activity", list(planner.ACTIVITY), index=list(planner.ACTIVITY).index(P["activity"]))
    P["goal"] = c3.selectbox("Goal", list(planner.GOALS), index=list(planner.GOALS).index(P["goal"]))
    store.save(data)
    if c2.button("📌 Log today's weight"):
        store.log_weight(data, float(P["weight"]))
        st.success("Weight logged — see the trend on the Dashboard.")
    t = planner.targets(P["age"], P["sex"], P["height"], P["weight"], P["activity"], P["goal"])
    mt = planner.macro_targets(t["kcal"], t["protein_g"])
    m = st.columns(6)
    m[0].metric("BMR (Mifflin–St Jeor)", f"{t['bmr']:.0f} kcal")
    m[1].metric("TDEE", f"{t['tdee']:.0f} kcal")
    m[2].metric("Daily target", f"{t['kcal']:.0f} kcal")
    m[3].metric("Protein", f"{mt['protein']:.0f} g")
    m[4].metric("Carbs", f"{mt['carbs']:.0f} g")
    m[5].metric("Fat", f"{mt['fat']:.0f} g")
    st.caption("BMR = 10·kg + 6.25·cm − 5·age + (5 male / −161 female). TDEE = BMR × activity factor. "
               "Target = TDEE − 500 (lose) / +300 (gain). Protein 1.2–1.8 g/kg by goal, fat 25 % of energy, carbs the rest.")

# ---------------------------------------------------------------- history tab
with tab_hist:
    if not data["meals"] and not data["workouts"]:
        st.info("Nothing logged yet.")
    else:
        rows = []
        for m in data["meals"]:
            for it in m["items"]:
                rows.append(dict(Date=m["date"], Time=m["time"], Meal=m["meal"], Food=it["food"].replace("_", " "),
                                 Grams=it["grams"], kcal=it["kcal"], Protein=it["protein"], Carbs=it["carbs"],
                                 Fat=it["fat"], Sample="yes" if m.get("demo") else ""))
        if rows:
            st.markdown("#### Meals")
            st.dataframe(pd.DataFrame(rows).sort_values(["Date", "Time"], ascending=False),
                         use_container_width=True, hide_index=True)
        if data["workouts"]:
            st.markdown("#### Workouts")
            st.dataframe(pd.DataFrame(data["workouts"]).sort_values("date", ascending=False),
                         use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear today's log"):
            for k in ("meals", "workouts"):
                data[k] = [x for x in data[k] if x["date"] != store.today()]
            store.save(data)
            st.rerun()
