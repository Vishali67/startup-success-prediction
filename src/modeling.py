import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE

import shap

data_path = "data/processed/engineered_startup_data.csv"
df = pd.read_csv(data_path)
print(f"✅ Loaded engineered data: {df.shape}")

# Convert important numeric columns safely
for col in ["funding_total_usd", "funding_rounds", "milestones", "relationships"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    else:
        df[col] = 0  # Create empty if missing

# Create a heuristic success indicator
df["status"] = np.where(
    (df["funding_total_usd"] > df["funding_total_usd"].median()) |
    (df["funding_rounds"] > df["funding_rounds"].median()) |
    (df["milestones"] > df["milestones"].median()) |
    (df["relationships"] > df["relationships"].median()),
    1,  # Successful
    0   # Unsuccessful
)

print("\n✅ Target distribution after recreation:")
print(df["status"].value_counts(normalize=True))


target_col = "status"
X = df.drop(columns=[target_col])
y = df[target_col]

# Drop non-numeric columns
X = X.select_dtypes(include=[np.number]).fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("\n✅ After SMOTE balancing:")
print(y_resampled.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
print(f"\n📊 Train: {X_train.shape} | Test: {X_test.shape}")

rf = RandomForestClassifier(n_estimators=200, random_state=42)
gb = GradientBoostingClassifier(random_state=42)
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5
)
lr = LogisticRegression(max_iter=500, random_state=42)

# Ensemble Voting Classifier
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("xgb", xgb), ("lr", lr)],
    voting="soft"
)

df = df[df["funding_rounds"] <= 1]
# Example (only for demonstration)
df.loc[df['milestones'] >= 3, 'success_label'] = 1
df.loc[(df['funding_total_usd'] > 1e7) & (df['funding_rounds_x'] > 3), 'success_label'] = 1

ensemble.fit(X_train, y_train)
print("\n✅ Model training complete!")

y_pred = ensemble.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

rf.fit(X_train, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[-15:]
plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), np.array(X.columns)[indices])
plt.title("Top 15 Feature Importances (RandomForest)")
plt.tight_layout()
plt.show()

os.makedirs("models", exist_ok=True)
joblib.dump(ensemble, "models/startup_success_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

feature_names = list(X.columns)
joblib.dump(feature_names, "models/feature_names.pkl")

print("\n💾 Model + Scaler saved in /models")

'''

# ==========================================
# modeling.py — Stage 2: Pre-Launch Behavior Focused Model
# ==========================================
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# Load engineered dataset
# ------------------------------------------------
data_path = "data/processed/engineered_startup_data.csv"
df = pd.read_csv(data_path)
print(f"✅ Loaded engineered data: {df.shape}")

# ------------------------------------------------
# Create smoother, more realistic target
# ------------------------------------------------

for col in ["funding_total_usd", "funding_rounds", "milestones", "relationships"]:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# A mild sigmoid-like transformation on funding (log scale)
df["funding_log"] = np.log1p(df["funding_total_usd"])

# Define success more continuously
success_score = (
    0.4 * (df["funding_log"] / df["funding_log"].max()) +
    0.2 * (df["funding_rounds"] / (df["funding_rounds"].max() + 1)) +
    0.2 * (df["milestones"] / (df["milestones"].max() + 1)) +
    0.2 * (df["relationships"] / (df["relationships"].max() + 1))
)

# Convert to binary success label with smoother threshold
df["status"] = (success_score > success_score.median()).astype(int)

print("\n✅ Target distribution after enhancement:")
print(df["status"].value_counts(normalize=True))

# ------------------------------------------------
# Feature preparation
# ------------------------------------------------
target_col = "status"
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode investor presence if available
for feat in ["has_angel", "has_venture", "has_roundA", "has_roundB"]:
    if feat not in X.columns:
        X[feat] = 0

# Interaction terms for realism
X["vc_funding_interaction"] = X["has_venture"] * df["funding_log"]
X["angel_milestone_interaction"] = X["has_angel"] * df["milestones"]

# Keep numeric only
X = X.select_dtypes(include=[np.number]).fillna(0)

# ------------------------------------------------
# Scaling and resampling
# ------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)
print("\n✅ After SMOTE balancing:")
print(y_res.value_counts())

# ------------------------------------------------
# Train / test split
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print(f"\n📊 Train: {X_train.shape} | Test: {X_test.shape}")

# ------------------------------------------------
# Models
# ------------------------------------------------
rf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
gb = GradientBoostingClassifier(random_state=42)
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
)
lr = LogisticRegression(max_iter=600, random_state=42, class_weight="balanced")

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("xgb", xgb), ("lr", lr)],
    voting="soft"
)

# ------------------------------------------------
# Train and evaluate
# ------------------------------------------------
ensemble.fit(X_train, y_train)
print("\n✅ Model training complete!")

y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {acc*100:.2f}%")

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix")
plt.show()

# ------------------------------------------------
# Save model and scaler
# ------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(ensemble, "models/startup_success_model_stage2.pkl")
joblib.dump(scaler, "models/scaler_stage2.pkl")
joblib.dump(list(X.columns), "models/feature_names_stage2.pkl")
print("\n💾 Saved improved model + scaler + features to /models/")
'''