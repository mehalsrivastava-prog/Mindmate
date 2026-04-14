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

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Match training features
    for col in feature_order:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_order]
    # clamp inputs to training range
    for key in ["academic_competition", "home_pressure", "peer_pressure"]:
     user_input[key] = max(1, min(5, float(user_input[key])))

    # 🔮 Prediction
    prediction = model.predict(input_df)[0]

    # 🔧 Manual correction (IMPORTANT PART)
    if user_input["coping_strategy"] == "Social support (friends, family)":
        prediction -= 0.4

    return {
        "score": round(float(prediction), 2),
        "level": interpret_stress(prediction)
    }