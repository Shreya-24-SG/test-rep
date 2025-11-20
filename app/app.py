from flask import Flask, render_template, request, make_response, redirect, url_for, session
import joblib
import pandas as pd
from xhtml2pdf import pisa
from io import BytesIO
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Load trained model
model = joblib.load("models/RandomForest.pkl")

# ------------------ LOGIN ------------------
users = {"doctor": "password123"}  # demo login

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in users and users[username] == password:
            session["user"] = username
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ------------------ DASHBOARD ------------------
@app.route("/")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

# ------------------ PREDICTION ------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    df_data = pd.read_csv("data/preterm_dataset.csv")
    features_sample = df_data.drop(columns=["Pre-term"]).iloc[0].to_dict()

    feature_ranges = {col: {"min": float(df_data[col].min()), "max": float(df_data[col].max())}
                      for col in df_data.drop(columns=["Pre-term"]).columns}

    low_risk_df = df_data[df_data["Pre-term"] == 0]
    ideal_ranges = {col: {"min": float(low_risk_df[col].min()), "max": float(low_risk_df[col].max())}
                    for col in df_data.drop(columns=["Pre-term"]).columns}

    if request.method == "POST":
        patient_keys = ["Patient_Name", "Age", "Weight", "Date"]
        patient_info = {key: request.form[key] for key in patient_keys}
        input_data = {key: [float(request.form[key])] for key in request.form.keys() if key not in patient_keys}
        df = pd.DataFrame(input_data)

        prediction_numeric = model.predict(df)[0]
        prob = float(model.predict_proba(df)[0][1])
        pred_label = "High Risk" if prediction_numeric == 1 else "Low Risk"
        color = "green" if prob < 0.3 else "orange" if prob < 0.7 else "red"

        patient_json = json.dumps(patient_info)

        return render_template(
            "result.html",
            pred=pred_label,
            prob=prob,
            patient=patient_info,
            color=color,
            patient_json=patient_json,
            features=input_data,
            ideal_ranges=ideal_ranges
        )

    return render_template("index.html", features=features_sample, feature_ranges=feature_ranges)

# ------------------ PDF GENERATION ------------------
def generate_pdf(template_src, context_dict):
    html = render_template(template_src, **context_dict)
    result = BytesIO()
    pdf = pisa.pisaDocument(BytesIO(html.encode("UTF-8")), result)
    if not pdf.err:
        return result.getvalue()
    return None

# Prediction report PDF
@app.route("/report_pdf", methods=["POST"])
def report_pdf():
    patient_info = json.loads(request.form.get("patient_json"))
    pred = request.form.get("pred")
    prob = float(request.form.get("prob"))
    features = json.loads(request.form.get("features"))
    ideal_ranges = json.loads(request.form.get("ideal_ranges"))

    context = {
        "patient": patient_info,
        "pred": pred,
        "prob": prob,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": features,
        "ideal_ranges": ideal_ranges
    }

    pdf_data = generate_pdf("report_template.html", context)
    response = make_response(pdf_data)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=preterm_report.pdf"
    return response

# Doctor-to-Patient Summary generator
def generate_patient_summary(pred, features, ideal_ranges):
    summary = ""
    if pred == "High Risk":
        summary += ("Based on your health data, there is a higher likelihood of preterm birth. "
                    "This does not mean it will definitely happen, but careful monitoring is essential.\n\n")
        # Risk factors
        risk_factors = []
        for f, vlist in features.items():
            val = float(vlist[0])
            min_val = float(ideal_ranges[f]["min"])
            max_val = float(ideal_ranges[f]["max"])
            if val < min_val or val > max_val:
                risk_factors.append(f"{f} (value: {val}, ideal: {min_val}-{max_val})")
        if risk_factors:
            summary += "Key risk factors: " + ", ".join(risk_factors) + ".\n\n"

        # Recommendations
        summary += (
            "Lifestyle & Support Recommendations:\n"
            "- Rest & Activity: Avoid strenuous activity; Bed rest if advised\n"
            "- Nutrition: High-protein, iron-rich diet; Prenatal vitamins and folic acid\n"
            "- Hydration: Adequate fluid intake\n"
            "- Mental Health: Counseling support; Stress management techniques\n"
            "- Avoid: Smoking, alcohol, and illicit drugs\n\n"
            "Delivery Planning:\n"
            "- Hospital Selection: Facility with NICU support\n"
            "- Delivery Mode: Vaginal or cesarean based on fetal and maternal condition\n"
            "- Timing: May require early induction or planned cesarean\n\n"
            "Emergency Signs to Watch For:\n"
            "- Vaginal bleeding\n- Severe abdominal pain\n- Blurred vision or severe headache\n"
            "- Fever >38.5°C\n- Sudden swelling of face/hands\n"
        )
    else:
        summary += ("Based on your health data, your risk of preterm birth is low. Continue routine prenatal care.\n\n")
        summary += "Lifestyle & Support Recommendations:\n"
        summary += "- Rest & Activity: Normal activity, moderate exercise\n"
        summary += "- Nutrition: Balanced diet with vitamins\n"
        summary += "- Hydration: Adequate fluid intake\n"
        summary += "- Mental Health: Relaxation techniques\n"
        summary += "- Avoid: Smoking, alcohol, illicit drugs\n\n"
        summary += "Delivery Planning:\n"
        summary += "- Hospital Selection: Any standard facility\n"
        summary += "- Delivery Mode: As advised by your doctor\n\n"
        summary += "Emergency Signs to Watch For:\n"
        summary += "- Vaginal bleeding, severe abdominal pain, blurred vision, fever >38.5°C, sudden swelling\n"

    summary += "\nAlways consult your doctor for concerns and follow medical guidance."
    return summary

@app.route("/patient_summary_pdf", methods=["POST"])
def patient_summary_pdf():
    if "user" not in session:
        return redirect(url_for("login"))

    patient_info = json.loads(request.form.get("patient_json"))
    pred = request.form.get("pred")
    features = json.loads(request.form.get("features"))
    ideal_ranges = json.loads(request.form.get("ideal_ranges"))

    summary = generate_patient_summary(pred, features, ideal_ranges)

    context = {
        "patient": patient_info,
        "pred": pred,
        "summary": summary,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": features,
        "ideal_ranges": ideal_ranges
    }

    pdf_data = generate_pdf("patient_summary_template.html", context)
    response = make_response(pdf_data)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename=patient_summary_{pred.replace(' ', '_')}.pdf"
    return response
# ------------------ NEW PREDICTION ------------------
@app.route("/predict_new", methods=["GET", "POST"])
def predict_new():
    if "user" not in session:
        return redirect(url_for("login"))

    df_data = pd.read_csv("data/dummy_preterm_data.csv")
    features_sample = df_data.drop(columns=["Preterm"]).iloc[0].to_dict()
    feature_ranges = {col: {"min": float(df_data[col].min()), "max": float(df_data[col].max())}
                      for col in df_data.drop(columns=["Preterm"]).columns}

    if request.method == "POST":
        patient_keys = ["Patient_Name", "Age", "Weight", "Date"]
        patient_info = {key: request.form[key] for key in patient_keys}

        # Load new model
        new_model = joblib.load("models/RandomForest_new.pkl")

        # Align input with new model’s trained feature names
        input_data = {key: float(request.form[key]) 
                      for key in new_model.feature_names_in_ if key in request.form}
        df = pd.DataFrame([input_data], columns=new_model.feature_names_in_)

        prediction_numeric = new_model.predict(df)[0]
        prob = float(new_model.predict_proba(df)[0][1])
        pred_label = "High Risk" if prediction_numeric == 1 else "Low Risk"
        color = "green" if prob < 0.3 else "orange" if prob < 0.7 else "red"

        patient_json = json.dumps(patient_info)

        return render_template(
            "result_new.html",
            pred=pred_label,
            prob=prob,
            patient=patient_info,
            color=color,
            patient_json=patient_json,
            features=input_data
        )

    return render_template("index_new.html", features=features_sample, feature_ranges=feature_ranges)
# ------------------ MAIN ------------------
if __name__ == "__main__":
    app.run(debug=True)
