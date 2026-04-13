import joblib
import pandas as pd
import numpy as np
import sys, json

# ---------------- INPUT ----------------
if len(sys.argv) > 1:
    input_data = json.loads(sys.argv[1])
else:
    # fallback for manual testing
    input_data = {}

# ---------------- LOAD MODELS ----------------
diet_model = joblib.load("diet_model.pkl")
diet_encoder = joblib.load("diet_encoder.pkl")

depression_model = joblib.load("depression_model.pkl")

financial_model = joblib.load("financial_stress_regressor.pkl")
academic_model = joblib.load("academic_stress_model.pkl")

master_model = joblib.load("master_model.pkl")

# academic feature order (IMPORTANT)
feature_order = joblib.load("feature_order.pkl")

# =====================================================
# 1️⃣ DIET MODEL
# =====================================================
diet_input = pd.DataFrame([{
    "meals": input_data.get("meals", 3),
    "junk_freq": input_data.get("junk_freq", 2),
    "fruits": input_data.get("fruits", 2),
    "vegetables": input_data.get("vegetables", 2),
    "fried": input_data.get("fried", 2),
    "water": input_data.get("water", 5),
    "BMI": input_data.get("BMI", 24),
    "sleep": input_data.get("sleep", 7),
    "exercise": input_data.get("exercise", 3),
    "calories": input_data.get("calories", 2000)
}])

diet_pred = diet_model.predict(diet_input)[0]
diet_label = diet_encoder.inverse_transform([diet_pred])[0]

diet_map = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}
diet_score = diet_map[diet_label] / 2


# =====================================================
# 2️⃣ DEPRESSION MODEL
# =====================================================
depression_input = pd.DataFrame([{
    "q1_sad": input_data.get("q1", 1),
    "q2_interest": input_data.get("q2", 1),
    "q3_sleep": input_data.get("q3", 1),
    "q4_energy": input_data.get("q4", 1),
    "q5_appetite": input_data.get("q5", 1),
    "q6_guilt": input_data.get("q6", 1),
    "q7_focus": input_data.get("q7", 1),
    "q8_movement": input_data.get("q8", 1),
    "q9_selfharm": input_data.get("q9", 0)
}])

proba = depression_model.predict_proba(depression_input)[0]
depression_score = proba[2]


# =====================================================
# 3️⃣ FINANCIAL MODEL
# =====================================================
financial_input = pd.DataFrame([{
    "FWB1_1": input_data.get("FWB1_1", 50),
    "FWB1_2": input_data.get("FWB1_2", 50),
    "FWB1_3": input_data.get("FWB1_3", 50),
    "SWB_1": input_data.get("SWB_1", 50),
    "SWB_2": input_data.get("SWB_2", 50),
    "SWB_3": input_data.get("SWB_3", 50),
    "fpl": input_data.get("fpl", 1)
}])

financial_pred = financial_model.predict(financial_input)[0]
financial_score = financial_pred / 100


# =====================================================
# 4️⃣ ACADEMIC MODEL (FIXED ORDER)
# =====================================================
academic_input = pd.DataFrame([{
    "academic_competition": input_data.get("academic_competition", 3),
    "coping_strategy": input_data.get("coping_strategy", 1),
    "study_environment": input_data.get("study_environment", 1),
    "home_pressure": input_data.get("home_pressure", 3),
    "peer_pressure": input_data.get("peer_pressure", 3)
}])

# ✅ ensure correct feature order
for col in feature_order:
    if col not in academic_input:
        academic_input[col] = 0

academic_input = academic_input[feature_order]

academic_pred = academic_model.predict(academic_input)[0]
academic_score = academic_pred / 5


# =====================================================
# 5️⃣ MASTER MODEL
# =====================================================
master_input = pd.DataFrame([{
    "diet_score": diet_score,
    "depression_score": depression_score,
    "financial_score": financial_score,
    "academic_score": academic_score
}])

final_pred = master_model.predict(master_input)[0]

labels = ["Low", "Moderate", "High"]


# =====================================================
# OUTPUT (ONLY JSON!)
# =====================================================
print(json.dumps({
    "prediction": labels[final_pred],
    "confidence": round(
        max(depression_score, academic_score, financial_score), 2
    )
}))