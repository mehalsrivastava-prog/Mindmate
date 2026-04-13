import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# ---------------- LOAD ----------------
df1 = pd.read_csv("Dietary Habits Survey Data.csv")
df2 = pd.read_csv("health_activity_data.csv")

df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# ---------------- OPTIONAL UPSAMPLING ----------------
df1 = df1.sample(500, replace=True, random_state=42).reset_index(drop=True)
df2 = df2.sample(500, replace=True, random_state=42).reset_index(drop=True)

# ---------------- RENAME ----------------
df1 = df1.rename(columns={
    "How many meals do you have a day? (number of regular occasions in a day when a significant and reasonably filling amount of food is eaten)": "meals",
    "How many times a week do you order-in or go out to eat?": "junk_freq",
    "What is your weekly food intake frequency of the following food categories: [Fresh fruit]": "fruits",
    "What is your weekly food intake frequency of the following food categories: [Fresh vegetables]": "vegetables",
    "What is your weekly food intake frequency of the following food categories: [Oily, fried foods]": "fried",
    "What is your water consumption like (in a day, 1 cup=250ml approx)": "water"
})

df2 = df2.rename(columns={
    "Hours_of_Sleep": "sleep",
    "Exercise_Hours_per_Week": "exercise",
    "Calories_Intake": "calories"
})

# ---------------- SELECT ----------------
df1 = df1[["meals", "junk_freq", "fruits", "vegetables", "fried", "water"]]
df2 = df2[["BMI", "sleep", "exercise", "calories"]]

# ---------------- HANDLE WATER ----------------
df1["water"] = pd.to_numeric(df1["water"], errors="coerce")
df1["water"] = df1["water"].fillna(df1["water"].mean())

# ---------------- MAP CATEGORICAL ----------------
mapping = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4
}

for col in ["fruits", "vegetables", "fried"]:
    df1[col] = df1[col].map(mapping)

df1.fillna(0, inplace=True)

# ---------------- MERGE ----------------
df2 = df2.sample(len(df1), replace=True).reset_index(drop=True)
combined = pd.concat([df1, df2], axis=1)
combined.fillna(0, inplace=True)

# ---------------- LABEL CREATION ----------------
def classify_diet(row):
    score = 0

    score += row["fruits"] * 2
    score += row["vegetables"] * 2
    score += row["water"] * 0.7
    score += row["exercise"] * 1.2

    if row["sleep"] >= 7:
        score += 2
    elif row["sleep"] < 5:
        score -= 2

    score -= row["fried"] * 1.5
    score -= row["junk_freq"] * 1.5

    if row["BMI"] > 30:
        score -= 3
    elif row["BMI"] > 25:
        score -= 1.5

    if row["calories"] > 3000:
        score -= 2
    elif row["calories"] < 1500:
        score -= 1

    if score >= 6:
        return "Healthy"
    elif score >= 2:
        return "Moderate"
    else:
        return "Unhealthy"

combined["Diet_Label"] = combined.apply(classify_diet, axis=1)

print("📊 Label Distribution:")
print(combined["Diet_Label"].value_counts())

# ---------------- ENCODE ----------------
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
combined["Diet_Label"] = le.fit_transform(combined["Diet_Label"])

# ---------------- TRAIN ----------------
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier   # ✅ CHANGED HERE

X = combined.drop("Diet_Label", axis=1)
y = combined["Diet_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print("\n📈 Test Accuracy:")
print(accuracy_score(y_test, y_test_pred))

print("\n📈 Train Accuracy:")
print(accuracy_score(y_train_pred, y_train))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))

# ---------------- SAVE ----------------
import joblib

joblib.dump(model, "diet_model.pkl")
joblib.dump(le, "diet_encoder.pkl")

print("\n✅ Decision Tree model trained successfully!")