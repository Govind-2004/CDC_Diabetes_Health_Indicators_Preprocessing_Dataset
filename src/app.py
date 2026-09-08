import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
)

# ==============================
# Load Model and Preprocessor
# ==============================

MODEL_PATH = BASE_DIR / "model" / "XGBoost.pkl"
PREPROCESSING_PATH = BASE_DIR / "model" / "preprocessing.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSING_PATH, "rb") as f:
    scaler = pickle.load(f)


# ==============================
# Model Feature Order
# ==============================

FEATURE_ORDER = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "AnyHealthcare",
    "GenHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income",
]


# ==============================
# Features that are scaled
# ==============================

SKEW_COLS = [
    "BMI",
    "GenHlth",
    "PhysHlth",
    "Age",
    "Education",
    "Income",
]


# ==============================
# Input validation bounds
# ==============================

FEATURE_BOUNDS = {
    "HighBP": (0, 1),
    "HighChol": (0, 1),
    "CholCheck": (0, 1),
    "BMI": (10, 90),
    "Smoker": (0, 1),
    "HeartDiseaseorAttack": (0, 1),
    "PhysActivity": (0, 1),
    "Fruits": (0, 1),
    "Veggies": (0, 1),
    "AnyHealthcare": (0, 1),
    "GenHlth": (1, 5),
    "PhysHlth": (0, 30),
    "DiffWalk": (0, 1),
    "Sex": (0, 1),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}


# ==============================
# Home page
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# Prediction
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "No input data received."
        }), 400

    # Check required fields
    missing = [feature for feature in FEATURE_ORDER if feature not in data]

    if missing:
        return jsonify({
            "error": f"Missing fields: {', '.join(missing)}"
        }), 400

    # Validate and collect input
    row = []

    for feature in FEATURE_ORDER:

        try:
            value = float(data[feature])

        except (TypeError, ValueError):

            return jsonify({
                "error": f"'{feature}' must be a number."
            }), 400

        lo, hi = FEATURE_BOUNDS[feature]

        if not (lo <= value <= hi):

            return jsonify({
                "error": f"'{feature}' must be between {lo} and {hi}."
            }), 400

        row.append(value)

    # Create DataFrame with exact feature order
    X = pd.DataFrame(
        [row],
        columns=FEATURE_ORDER
    )

    # Apply the fitted MinMaxScaler
    X[SKEW_COLS] = scaler.transform(
        X[SKEW_COLS]
    )

    # Prediction probabilities
    probs = model.predict_proba(X)[0]

    # [P(no diabetes), P(predibetes), P(diabetes)]
    p_no_risk = float(probs[0])
    p_at_risk = float(probs[1] + probs[2])

    result = "at_risk" if p_at_risk >= 0.5 else "no_risk"

    return jsonify({
        "result": result,
        "probability_at_risk": round(p_at_risk, 4),
        "probability_no_risk": round(p_no_risk, 4),
    })


# ==============================
# Run Flask
# ==============================

if __name__ == "__main__":
    app.run(debug=True)