import sys
import json
import numpy as np
import joblib

try:
    # ---------------- LOAD MODEL ----------------
    model = joblib.load("diet_model.pkl")

    # ---------------- INPUT ----------------
    input_data = json.loads(sys.argv[1])

    # ---------------- MAPPING ----------------
    mapping = {
        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        "Always": 4
    }

    # ---------------- EXTRACT FEATURES ----------------
    meals = float(input_data.get("meals", 0))
    junk_freq = float(input_data.get("junk_freq", 0))
    fruits = mapping.get(input_data.get("fruits"), 2)
    vegetables = mapping.get(input_data.get("vegetables"), 2)
    fried = mapping.get(input_data.get("fried"), 2)
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

    # ---------------- MAP LABEL ----------------
    label_map = {
        0: "Unhealthy",
        1: "Moderate",
        2: "Healthy"
    }

    label = label_map.get(pred, "Moderate")

    # ---------------- OUTPUT ----------------
    print(json.dumps({
        "prediction": label
    }))

except Exception as e:
    print(json.dumps({
        "error": str(e)
    }))