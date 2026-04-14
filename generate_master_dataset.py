import numpy as np
import pandas as pd

np.random.seed(42)
n = 4000

data = []

for _ in range(n):

    # ---------------- FORCE BALANCED LABEL ----------------
    rand = np.random.rand()

    if rand < 0.33:
        label = 0   # LOW

        diet_score = np.random.uniform(0.7, 1.0)
        depression_score = np.random.uniform(0.0, 0.3)
        financial_score = np.random.uniform(0.0, 0.4)
        academic_score = np.random.uniform(0.0, 0.4)

    elif rand < 0.66:
        label = 1   # MODERATE

        diet_score = np.random.uniform(0.3, 0.7)
        depression_score = np.random.uniform(0.3, 0.7)
        financial_score = np.random.uniform(0.3, 0.7)
        academic_score = np.random.uniform(0.3, 0.7)

    else:
        label = 2   # HIGH

        diet_score = np.random.uniform(0.0, 0.4)
        depression_score = np.random.uniform(0.6, 1.0)
        financial_score = np.random.uniform(0.6, 1.0)
        academic_score = np.random.uniform(0.6, 1.0)

    # ---------------- ADD SMALL NOISE (REALISM) ----------------
    diet_score = np.clip(diet_score + np.random.normal(0, 0.05), 0, 1)
    depression_score = np.clip(depression_score + np.random.normal(0, 0.05), 0, 1)
    financial_score = np.clip(financial_score + np.random.normal(0, 0.05), 0, 1)
    academic_score = np.clip(academic_score + np.random.normal(0, 0.05), 0, 1)

    data.append([
        diet_score,
        depression_score,
        financial_score,
        academic_score,
        label
    ])

# ---------------- CREATE DATAFRAME ----------------
df = pd.DataFrame(data, columns=[
    "diet_score",
    "depression_score",
    "financial_score",
    "academic_score",
    "target"
])

# ---------------- CHECK BALANCE ----------------
print("\n📊 Class Distribution:")
print(df["target"].value_counts())

# ---------------- SAVE ----------------
df.to_csv("master_dataset.csv", index=False)

print("\n✅ Balanced dataset generated successfully!")