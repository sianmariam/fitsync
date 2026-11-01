"""Calorie targets (BMR/TDEE) and a rule-based workout planner linked to the food log."""

ACTIVITY = {"Sedentary": 1.2, "Lightly active": 1.375, "Moderately active": 1.55, "Very active": 1.725}
GOALS = {"Lose weight": -500, "Maintain": 0, "Gain muscle": +300}
PROTEIN_PER_KG = {"Lose weight": 1.6, "Maintain": 1.2, "Gain muscle": 1.8}
# MET values (Compendium of Physical Activities, approx.)
METS = {"Brisk walking": 4.3, "Jogging": 7.0, "Cycling (moderate)": 6.8, "Skipping rope": 11.0}


def targets(age, sex, height, weight, activity, goal):
    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if sex == "Male" else -161)
    tdee = bmr * ACTIVITY[activity]
    return dict(bmr=bmr, tdee=tdee, kcal=max(1200, tdee + GOALS[goal]), protein_g=PROTEIN_PER_KG[goal] * weight)


def burn_minutes(kcal, weight):
    # kcal/min = MET * 3.5 * kg / 200
    return {n: kcal / (m * 3.5 * weight / 200) for n, m in METS.items()}


LIB = {
    "None (home)": {"push": ["Push-ups", "Pike push-ups", "Bench dips"], "pull": ["Superman hold", "Towel rows"],
                    "legs": ["Squats", "Lunges", "Glute bridges"], "core": ["Plank", "Bicycle crunches"],
                    "cardio": ["Jumping jacks", "High knees", "Burpees"]},
    "Dumbbells": {"push": ["DB bench press", "DB shoulder press", "DB triceps extension"],
                  "pull": ["DB bent-over row", "DB biceps curl", "DB reverse fly"],
                  "legs": ["Goblet squat", "DB Romanian deadlift", "DB lunges"], "core": ["Plank", "DB Russian twist"],
                  "cardio": ["DB thrusters", "Skipping", "Mountain climbers"]},
    "Full gym": {"push": ["Bench press", "Overhead press", "Cable triceps pushdown"],
                 "pull": ["Lat pulldown", "Seated cable row", "Barbell curl"],
                 "legs": ["Back squat", "Leg press", "Romanian deadlift"], "core": ["Cable crunch", "Hanging knee raise"],
                 "cardio": ["Treadmill intervals", "Rowing machine", "Stair climber"]},
}
SPLITS = {2: [["push", "pull", "core"], ["legs", "cardio"]],
          3: [["push", "core"], ["pull", "cardio"], ["legs", "core"]],
          4: [["push"], ["legs"], ["pull", "core"], ["cardio", "core"]],
          5: [["push"], ["pull"], ["legs"], ["cardio", "core"], ["push", "pull"]],
          6: [["push"], ["pull"], ["legs"], ["cardio"], ["push", "core"], ["legs", "pull"]]}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def workout_plan(goal, days, level, equip):
    sets = {"Beginner": 3, "Intermediate": 4, "Advanced": 5}[level]
    reps = {"Lose weight": "12–15", "Maintain": "10–12", "Gain muscle": "6–10"}[goal]
    rest = {"Lose weight": "30–45 s", "Maintain": "60 s", "Gain muscle": "90–120 s"}[goal]
    plan = []
    for i, groups in enumerate(SPLITS[days]):
        ex = []
        for g in groups:
            for name in LIB[equip][g][: 3 if len(groups) == 1 else 2]:
                if g == "cardio":
                    ex.append(dict(Exercise=name, Sets=sets, Reps="40 s on / 20 s off", Rest="20 s"))
                elif g == "core" and "Plank" in name:
                    ex.append(dict(Exercise=name, Sets=sets, Reps="30–60 s hold", Rest="30 s"))
                else:
                    ex.append(dict(Exercise=name, Sets=sets, Reps=reps, Rest=rest))
        if goal == "Lose weight":
            ex.append(dict(Exercise="Finisher: brisk walk / cycle", Sets=1, Reps="20 min", Rest="—"))
        plan.append(dict(day=DAYS[i], focus=" + ".join(g.title() for g in groups), exercises=ex))
    return plan


def macro_targets(kcal, protein_g):
    """Protein from goal (g/kg); fat 25% of energy; carbs fill the rest."""
    fat = 0.25 * kcal / 9
    carbs = max(0.0, (kcal - protein_g * 4 - fat * 9) / 4)
    return dict(protein=protein_g, carbs=carbs, fat=fat)


def exercise_minutes_kcal(ex, weight):
    """Rough duration + energy for one planned exercise (MET-based)."""
    sets = int(ex.get("Sets", 3))
    name = ex["Exercise"].lower()
    if "finisher" in name:
        mins, met = 20, 5.0
    elif "40 s on" in str(ex.get("Reps", "")):
        mins, met = sets * 1.0, 8.0
    elif "hold" in str(ex.get("Reps", "")):
        mins, met = sets * 1.0, 3.8
    else:
        mins, met = sets * 2.0, 5.0
    return mins, met * 3.5 * weight / 200 * mins


def todays_session(goal, days, level, equip, weekday=None):
    """Map today's weekday onto the plan (Mon..Sat). Sunday = rest."""
    import datetime as dt
    wd = dt.date.today().weekday() if weekday is None else weekday
    plan = workout_plan(goal, days, level, equip)
    day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][wd]
    return next((p for p in plan if p["day"] == day_name), None), day_name


def bmi(weight, height):
    b = weight / (height / 100) ** 2
    cat = "Underweight" if b < 18.5 else "Normal" if b < 23 else "Overweight" if b < 25 else "Obese"  # Asian cut-offs
    return b, cat
