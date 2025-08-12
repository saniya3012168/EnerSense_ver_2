from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, flash
from enersense import (
    generate_synthetic_data,
    train_consumption_model,
    forecast_solar_generation
)
from agent import run_energy_agent  # optional; kept for backwards compatibility
import datetime, io, csv, os, pandas as pd
import numpy as np
import logging

# ---------------- App init ----------------
app = Flask(__name__)
app.secret_key = "change-me-to-a-secure-random-key"
app.logger.setLevel(logging.INFO)

# Data folders & small persistence
DATA_DIR = "data"
FEEDBACK_CSV = os.path.join(DATA_DIR, "feedback.csv")
COMMUNITIES_CSV = os.path.join(DATA_DIR, "communities.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- Model setup ----------------
data = generate_synthetic_data()
model = train_consumption_model(data)

# In-memory logs
prediction_logs = []

# ---------------- Helpers ----------------
def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def save_feedback_row(row):
    header = ["timestamp", "predicted_kWh", "actual_kWh", "rating", "notes"]
    exists = os.path.exists(FEEDBACK_CSV)
    with open(FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def read_feedback():
    if not os.path.exists(FEEDBACK_CSV):
        return []
    return pd.read_csv(FEEDBACK_CSV).to_dict(orient="records")

# ---------------- Input Agent (validation) ----------------
def validate_input(form):
    errors = []
    # required numeric fields
    try:
        t = float(form.get("temperature", ""))
    except:
        errors.append("Temperature must be a number.")
    try:
        h = float(form.get("humidity", ""))
    except:
        errors.append("Humidity must be a number.")
    try:
        s = float(form.get("solar", ""))
    except:
        errors.append("Solar must be a number.")
    try:
        a = int(form.get("appliances", ""))
    except:
        errors.append("Appliances must be an integer.")
    try:
        inc = float(form.get("income", ""))
    except:
        errors.append("Income must be a number.")
    return errors

# ---------------- Prediction Agent ----------------
def predict_from_form(form):
    # Build one-row dataframe as enersense expects (use data sample row)
    df = data.iloc[[0]].copy()
    df["Temperature"] = safe_float(form.get("temperature"), df["Temperature"].iloc[0])
    df["Humidity"] = safe_float(form.get("humidity"), df["Humidity"].iloc[0])
    df["Solar"] = safe_float(form.get("solar"), df["Solar"].iloc[0])
    df["Appliances"] = int(form.get("appliances", df["Appliances"].iloc[0]))
    df["Income"] = safe_float(form.get("income"), df["Income"].iloc[0])
    pred = model.predict(df[["Temperature", "Humidity", "Solar", "Appliances", "Income"]])[0]
    return round(float(pred), 3), df

# ---------------- Billing Agent ----------------
def estimate_bill(kwh, tariff_per_kwh=8.0):
    daily = round(kwh * tariff_per_kwh, 2)
    weekly = round(daily * 7, 2)
    monthly = round(daily * 30, 2)
    return {"daily": daily, "weekly": weekly, "monthly": monthly, "tariff": tariff_per_kwh}

# ---------------- Optimization Agent (time slot suggestions) ----------------
def time_slot_recommendations(solar_value, tariff_schedule=None):
    # tariff_schedule: dict of hour->tariff (optional). We'll return recommended hours.
    # Simple heuristic: if solar_value high -> recommend midday (10-16), if low -> recommend off-peak tariff hours if provided
    rec = []
    if solar_value >= 500:
        rec.append("Prefer running heavy appliances between 10:00 - 16:00 (high solar generation).")
    else:
        rec.append("Solar is low — consider shifting flexible loads to off-peak tariff hours if available.")
    if tariff_schedule:
        # find hours with min tariff
        min_tariff = min(tariff_schedule.values())
        off_peak_hours = [h for h,t in tariff_schedule.items() if t == min_tariff]
        rec.append(f"Off-peak tariff hours: {off_peak_hours}")
    return rec

# ---------------- Climate Agent ----------------
def climate_adjustment_prediction(prediction, temperature, humidity):
    # small rule: if temperature very high, extra cooling load increases consumption
    adj = prediction
    notes = []
    if temperature >= 30:
        adj *= 1.08  # +8%
        notes.append("High temperature → expect higher cooling load (+8%).")
    if humidity >= 80:
        adj *= 1.03
        notes.append("High humidity → slight increase in energy use (+3%).")
    return round(adj, 3), notes

# ---------------- Recommendation Agent ----------------
def generate_recommendations(log):
    recs = []
    if not log:
        recs.append("No data yet — make a prediction to receive recommendations.")
        return recs
    kwh = log.get("predicted_kWh", 0)
    appliances = log.get("appliances", 1)
    solar = log.get("solar", 0)
    income = log.get("income", 0)

    if kwh > 10:
        recs.append("High predicted consumption — consider rooftop solar (3-5 kW) + small battery 5-10 kWh.")
    if appliances >= 6:
        recs.append("Many appliances detected — use smart plugs and stagger start times.")
    if solar >= 400:
        recs.append("Good solar irradiance — consider investing in panels and time-shifting heavy loads to midday.")
    if income and income < 40000:
        recs.append("Check community or subsidy programs for clean energy finance.")
    # general tips
    recs.extend([
        "Use LED lighting and energy-efficient appliances.",
        "Run washing/dishwashing during daylight if you have solar.",
        "Lower AC thermostat by 1-2°C and use fans to reduce consumption."
    ])
    # keep unique
    return list(dict.fromkeys(recs))

# ---------------- Learning Agent ----------------
def learning_insights():
    feedback = read_feedback()
    if not feedback:
        return {"count": 0, "avg_error_kwh": None}
    # compute avg absolute error between predicted and actual where actual present
    df = pd.DataFrame(feedback)
    df = df[df["actual_kWh"].notnull() & (df["actual_kWh"] != "")]
    if df.empty:
        return {"count": len(feedback), "avg_error_kwh": None}
    df["error"] = (df["predicted_kWh"].astype(float) - df["actual_kWh"].astype(float)).abs()
    return {"count": len(feedback), "avg_error_kwh": round(df["error"].mean(), 3)}

# ---------------- Routes ----------------

@app.route('/')
def home():
    return render_template("index.html")

# enhanced predict route uses Input Agent, Prediction Agent, Climate Agent, Recommendation Agent, and logs
@app.route('/predict', methods=['POST'])
def predict():
    # Input Agent validation
    errors = validate_input(request.form)
    if errors:
        for e in errors:
            flash(e, "danger")
        return redirect(url_for("home"))

    predicted_kwh, df = predict_from_form(request.form)

    # Climate agent adjusts prediction
    adjusted, climate_notes = climate_adjustment_prediction(predicted_kwh,
                                                           float(request.form.get("temperature", 0)),
                                                           float(request.form.get("humidity", 0)))

    # Log entry
    log = {
        "timestamp": str(datetime.datetime.now()),
        "temperature": float(request.form.get("temperature", df["Temperature"].iloc[0])),
        "humidity": float(request.form.get("humidity", df["Humidity"].iloc[0])),
        "solar": float(request.form.get("solar", df["Solar"].iloc[0])),
        "appliances": int(request.form.get("appliances", df["Appliances"].iloc[0])),
        "income": float(request.form.get("income", df["Income"].iloc[0])),
        "predicted_kWh": round(predicted_kwh, 3),
        "adjusted_kWh": round(adjusted, 3),
        "climate_notes": "; ".join(climate_notes),
        "actual_kWh": ""
    }
    prediction_logs.append(log)

    # Recommendation agent
    recs = generate_recommendations(log)

    # show results (prediction + recommendations + quick billing)
    bill = estimate_bill(adjusted)
    return render_template("index.html", result=round(adjusted, 3), recommendations=recs, bill=bill)

# Billing page
@app.route('/billing')
def billing():
    if not prediction_logs:
        flash("No prediction logs yet.", "warning")
        return redirect(url_for("home"))
    last = prediction_logs[-1]
    kwh = last.get("adjusted_kWh", last.get("predicted_kWh", 0))
    bill = estimate_bill(kwh)
    return render_template("billing.html", bill=bill, log=last)

# Grid optimization (existing)
@app.route('/grid-opt')
def grid_opt():
    if not prediction_logs:
        return render_template("grid_optimization.html", optimized=None, msg="No prediction available.")
    latest = prediction_logs[-1]
    demand = float(latest.get("adjusted_kWh", latest.get("predicted_kWh", 0)))
    solar_kwh = forecast_solar_generation(latest["solar"])
    # improved optimize_grid uses simple storage defaults
    from_result = optimize_grid(demand, solar_kwh, storage_capacity_kwh=12, storage_current_kwh=3)
    # also suggest time slots
    time_recs = time_slot_recommendations(latest["solar"])
    return render_template("grid_optimization.html", optimized=from_result, solar_kwh=round(solar_kwh, 3),
                           demand=demand, time_recs=time_recs)

# Equity (existing)
@app.route('/equity')
def equity():
    if not prediction_logs:
        return render_template("equity_dashboard.html", allocations=None, msg="No prediction history.")
    recent = prediction_logs[-10:] if len(prediction_logs) >= 10 else prediction_logs
    available = sum([r["solar"] * 0.15 * 10 / 1000 for r in recent])
    # load communities (create default if missing)
    if not os.path.exists(COMMUNITIES_CSV):
        pd.DataFrame([
            {"community": "NorthTown", "population": 1200, "priority_score": 0.7},
            {"community": "EastVille", "population": 500, "priority_score": 0.9},
            {"community": "SouthPark", "population": 900, "priority_score": 0.6},
            {"community": "WestSide", "population": 400, "priority_score": 0.8}
        ]).to_csv(COMMUNITIES_CSV, index=False)
    communities = pd.read_csv(COMMUNITIES_CSV)
    # equity allocation using earlier function
    alloc_df = allocate_equity(available, communities)
    return render_template("equity_dashboard.html", allocations=alloc_df.to_dict(orient="records"), available=round(available, 3))

# Recommendations page
@app.route('/recommendations')
def recommendations_page():
    latest = prediction_logs[-1] if prediction_logs else None
    recs = generate_recommendations(latest)
    return render_template("clean_energy.html", recommendations=recs, latest=latest)

# Feedback (Learning Agent) submit actual vs predicted and rating
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == "POST":
        predicted = request.form.get("predicted_kWh")
        actual = request.form.get("actual_kWh")
        rating = request.form.get("rating")
        notes = request.form.get("notes", "")
        row = {
            "timestamp": str(datetime.datetime.now()),
            "predicted_kWh": predicted,
            "actual_kWh": actual,
            "rating": rating,
            "notes": notes
        }
        save_feedback_row(row)
        flash("Thank you for feedback — it has been saved.", "success")
        return redirect(url_for("home"))
    # GET => show small feedback page
    insights = learning_insights()
    return render_template("feedback.html", insights=insights)

# Learning insights (for admin)
@app.route('/learning-insights')
def learning_insights_page():
    insights = learning_insights()
    samples = read_feedback()
    return render_template("learning_insights.html", insights=insights, samples=samples)

# Charts & compare & others (unchanged)
@app.route('/charts')
def charts():
    timestamps = [r['timestamp'] for r in prediction_logs]
    predictions = [float(r['adjusted_kWh'] if r.get("adjusted_kWh") else r['predicted_kWh']) for r in prediction_logs]
    return render_template("chart_dashboard.html", timestamps=timestamps, predictions=predictions)

@app.route('/compare')
def compare():
    timestamps = [r['timestamp'] for r in prediction_logs]
    predictions = [float(r['predicted_kWh']) for r in prediction_logs]
    actuals = [float(r.get('actual_kWh') or 0) for r in prediction_logs]
    return render_template("compare_chart.html", timestamps=timestamps, predictions=predictions, actuals=actuals)

# Simple download (CSV)
@app.route('/download')
def download():
    if not prediction_logs:
        return "No data to download", 404
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=prediction_logs[0].keys())
    writer.writeheader()
    writer.writerows(prediction_logs)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv',
                     download_name='prediction_logs.csv', as_attachment=True)

# Agent decision (keeps previous capability)
@app.route('/agent-decision', methods=['POST'])
def agent_decision():
    input_data = request.get_json()
    result = run_energy_agent(input_data) if 'run_energy_agent' in globals() else {"status": "agent function not available"}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
