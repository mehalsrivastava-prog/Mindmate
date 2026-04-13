import joblib
import pandas as pd

def interpret_stress(score):
    if score < 2:
        return "Low"
    elif score < 3.5:
        return "Moderate"
    else:
        return "High"

model = joblib.load("academic_stress_model.pkl")
feature_order = joblib.load("feature_order.pkl")

def predict_academic_stress(user_input):

    input_df = pd.DataFrame([user_input])

    coping_map = {"Healthy": 2, "Moderate": 1, "Unhealthy": 0}
    env_map = {"Good": 2, "Average": 1, "Poor": 0}

    input_df["coping_strategy"] = input_df["coping_strategy"].map(coping_map)
    input_df["study_environment"] = input_df["study_environment"].map(env_map)

    for col in feature_order:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_order]

    prediction = model.predict(input_df)[0]

    return {
        "score": float(prediction),
        "level": interpret_stress(prediction)
    }