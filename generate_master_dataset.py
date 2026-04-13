import numpy as np
import pandas as pd

np.random.seed(42)
n = 3000

data = []

for _ in range(n):

    # ---------------- DIET SCORE (0–1) ----------------
    diet_score = np.random.beta(2, 2)  # realistic distribution

    # ---------------- DEPRESSION SCORE (0–1) ----------------
    # worse diet → higher depression
    depression_score = np.clip(
        (1 - diet_score) * 0.6 + np.random.normal(0.2, 0.15),
        0, 1
    )

    # ---------------- FINANCIAL SCORE (0–1) ----------------
    financial_score = np.clip(np.random.normal(0.5, 0.2), 0, 1)

    # ---------------- ACADEMIC SCORE (0–1) ----------------
    academic_score = np.clip(np.random.normal(0.6, 0.2), 0, 1)

    # ---------------- FINAL STRESS SCORE ----------------
    final_score = (
        (1 - diet_score) * 0.25 +
        depression_score * 0.35 +
        financial_score * 0.2 +
        academic_score * 0.2 +
        np.random.normal(0, 0.05)
    )

    # ---------------- LABEL ----------------
    if final_score < 0.33:
        label = 0  # Low
    elif final_score < 0.66:
        label = 1  # Moderate
    else:
        label = 2  # High

    data.append([
        diet_score,
        depression_score,
        financial_score,
        academic_score,
        label
    ])

df = pd.DataFrame(data, columns=[
    "diet_score",
    "depression_score",
    "financial_score",
    "academic_score",
    "target"
])

df.to_csv("master_dataset.csv", index=False)

print("✅ Dataset created")
print(df.head())
print(df["target"].value_counts())