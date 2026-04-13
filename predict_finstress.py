import sys
import json
import joblib
import numpy as np

model = joblib.load("financial_stress_regressor.pkl")

data = json.loads(sys.argv[1])

features = np.array([[
    data["FWB1_1"],
    data["FWB1_2"],
    data["FWB1_3"],
    data["SWB_1"],
    data["SWB_2"],
    data["SWB_3"],
    data["fpl"]
]])

prediction = model.predict(features)[0]

# ✅ ONLY JSON OUTPUT
print(json.dumps({
    "financialStress": round(float(prediction), 2)
}))