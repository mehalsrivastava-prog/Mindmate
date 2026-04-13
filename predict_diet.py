import sys
import json
import numpy as np
import joblib
import traceback

try:
    # ---------------- LOAD ----------------
    model = joblib.load("diet_model.pkl")
    le = joblib.load("diet_encoder.pkl")

    # ---------------- INPUT ----------------
    input_data = json.loads(sys.argv[1])

    mapping = {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        "Always": 4
    }

    # ---------------- EXTRACT ----------------
    meals = float(input_data.get("meals", 0))
    junk_freq = float(input_data.get("junk_freq", 0))
    fruits = mapping.get(input_data.get("fruits"), 0)
    vegetables = mapping.get(input_data.get("vegetables"), 0)
    fried = mapping.get(input_data.get("fried"), 0)
    water = float(input_data.get("water", 0))
    BMI = float(input_data.get("BMI", 0))
    sleep = float(input_data.get("sleep", 0))
    exercise = float(input_data.get("exercise", 0))
    calories = float(input_data.get("calories", 0))

    # ---------------- ARRAY ----------------
    input_array = np.array([[
        meals, junk_freq, fruits, vegetables, fried,
        water, BMI, sleep, exercise, calories
    ]])

    # ---------------- PREDICT ----------------
    pred = model.predict(input_array)[0]
    label = le.inverse_transform([pred])[0]

    # ---------------- OUTPUT ----------------
    result = {
        "prediction": label
    }

    print(json.dumps(result))

except Exception:
    print(traceback.format_exc())