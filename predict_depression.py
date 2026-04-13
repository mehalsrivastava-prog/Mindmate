import sys, json, joblib
import numpy as np

model = joblib.load("depression_model.pkl")
le = joblib.load("depression_encoder.pkl")

data = json.loads(sys.argv[1])

features = np.array([[
    data["q1"], data["q2"], data["q3"],
    data["q4"], data["q5"], data["q6"],
    data["q7"], data["q8"], data["q9"]
]])

pred = model.predict(features)[0]
proba = model.predict_proba(features).max()

label = le.inverse_transform([pred])[0]

print(json.dumps({
    "prediction": label,
    "confidence": round(proba * 100, 2)
}))