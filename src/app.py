import pickle
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root, one level above src/

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
)

MODEL_PATH = BASE_DIR / "model" / "XGBoost.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Exact order the model was trained on (from model.feature_names_in_)
FEATURE_ORDER = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

# Reasonable bounds per BRFSS coding, used to reject garbage input server-side
FEATURE_BOUNDS = {
    "HighBP": (0, 1), "HighChol": (0, 1), "CholCheck": (0, 1),
    "BMI": (10, 90), "Smoker": (0, 1), "Stroke": (0, 1),
    "HeartDiseaseorAttack": (0, 1), "PhysActivity": (0, 1), "Fruits": (0, 1),
    "Veggies": (0, 1), "HvyAlcoholConsump": (0, 1), "AnyHealthcare": (0, 1),
    "NoDocbcCost": (0, 1), "GenHlth": (1, 5), "MentHlth": (0, 30),
    "PhysHlth": (0, 30), "DiffWalk": (0, 1), "Sex": (0, 1), "Age": (1, 13),
    "Education": (1, 6), "Income": (1, 8),
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No input data received."}), 400

    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    row = []
    for feature in FEATURE_ORDER:
        try:
            value = float(data[feature])
        except (TypeError, ValueError):
            return jsonify({"error": f"'{feature}' must be a number."}), 400
        lo, hi = FEATURE_BOUNDS[feature]
        if not (lo <= value <= hi):
            return jsonify({"error": f"'{feature}' must be between {lo} and {hi}."}), 400
        row.append(value)

    X = np.array([row])
    probs = model.predict_proba(X)[0]  # [P(no diabetes), P(prediabetes), P(diabetes)]

    p_no_risk = float(probs[0])
    p_at_risk = float(probs[1] + probs[2])  # merge prediabetes + diabetes classes
    result = "at_risk" if p_at_risk >= 0.5 else "no_risk"

    return jsonify({
        "result": result,
        "probability_at_risk": round(p_at_risk, 4),
        "probability_no_risk": round(p_no_risk, 4),
    })


if __name__ == "__main__":
    app.run(debug=True)
