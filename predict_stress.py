import sys
import json
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Read input
input_data = json.loads(sys.argv[1])

features = np.array([[
    input_data["age"],
    input_data["sleep"],
    input_data["work_hours"],
    input_data["activity"],
    input_data["social"],
    input_data["stress_self"]
]])

# Scale
features_scaled = scaler.transform(features)

# Predict
prediction = model.predict(features_scaled)[0]
proba = model.predict_proba(features_scaled)[0]
confidence = int(max(proba) * 100)

# Explanation
explanation = f"Your stress level is predicted as {prediction} based on sleep, work hours, activity, and stress."

# Tips
tips_map = {
    "Low": [
        "Great job maintaining balance!",
        "Keep up your healthy routine."
    ],
    "Medium": [
        "Try improving your sleep schedule.",
        "Take breaks during work.",
        "Stay socially active."
    ],
    "High": [
        "Practice relaxation techniques (deep breathing).",
        "Talk to someone you trust.",
        "Reduce workload if possible."
    ]
}

output = {
    "prediction": prediction,
    "confidence": confidence,
    "explanation": explanation,
    "tips": tips_map.get(prediction, [])
}

print(json.dumps(output))