import joblib
import pandas as pd

def interpret_stress(score):
    if score < 2:
        return "Low"
    elif score < 3.5:
        return "Moderate"
    else:
        return "High"

# LOAD FILES
model = joblib.load("academic_stress_model.pkl")
le_coping = joblib.load("coping_encoder.pkl")
le_env = joblib.load("env_encoder.pkl")
feature_order = joblib.load("feature_order.pkl")

def predict_academic_stress(user_input):
    input_df = pd.DataFrame([user_input])
    print("INPUT DF:", input_df)  
    try:
        input_df["coping_strategy"] = le_coping.transform(input_df["coping_strategy"])
        input_df["study_environment"] = le_env.transform(input_df["study_environment"])
    except:
        return {"error": "Invalid category input"}
    for col in feature_order:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[feature_order]

    prediction = model.predict(input_df)[0]
    print("COLUMNS:", input_df.columns)
    print("VALUES:", input_df.values)
    return {
        "score": float(prediction),
        "level": interpret_stress(prediction)
    }