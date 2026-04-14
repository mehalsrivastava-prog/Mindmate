import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ================= LOAD DATA =================
df = pd.read_csv("diet_dataset.csv")

print("📊 Label Distribution:")
print(df["Diet_Label"].value_counts())

# ================= FEATURES & TARGET =================
X = df.drop("Diet_Label", axis=1)
y = df["Diet_Label"]

# ================= ENCODE LABEL =================
le = LabelEncoder()
y = le.fit_transform(y)

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= MODEL =================
model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ================= EVALUATION =================
y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print("\n📈 Test Accuracy:")
print(accuracy_score(y_test, y_test_pred))

print("\n📈 Train Accuracy:")
print(accuracy_score(y_train, y_train_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))

# ================= SAVE =================
joblib.dump(model, "diet_model.pkl")
joblib.dump(le, "diet_encoder.pkl")

print("\n✅ Diet model trained successfully!")