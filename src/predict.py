'''import pandas as pd
import numpy as np
import joblib
import os

# ===============================
# Load Model & Scaler
# ===============================
print("✅ Loading model, scaler, and feature names...")

model = joblib.load("models/startup_success_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# ===============================
# Load New Data
# ===============================
data_path = "data/processed/engineered_startup_data.csv"
df_new = pd.read_csv(data_path)
print(f"📄 Loaded new data: {df_new.shape}")

# ===============================
# Prepare Features (Align Columns)
# ===============================
# Drop any target columns if exist
for col in ["status", "success_label"]:
    if col in df_new.columns:
        df_new = df_new.drop(columns=[col])

# Keep only numeric columns
numeric_df = df_new.select_dtypes(include=[np.number]).fillna(0)

# Align with training feature names
missing_cols = [c for c in feature_names if c not in numeric_df.columns]
extra_cols = [c for c in numeric_df.columns if c not in feature_names]

if missing_cols:
    print(f"⚠️ Adding {len(missing_cols)} missing columns with 0s: {missing_cols}")
    for c in missing_cols:
        numeric_df[c] = 0

if extra_cols:
    print(f"⚠️ Dropping {len(extra_cols)} extra columns: {extra_cols}")
    numeric_df = numeric_df.drop(columns=extra_cols)

# Reorder columns to match training
numeric_df = numeric_df[feature_names]

# ===============================
# Scale and Predict
# ===============================
X_scaled = scaler.transform(numeric_df)
preds = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)[:, 1]

# ===============================
# Results
# ===============================
if "name" in df_new.columns:
    results = pd.DataFrame({
        "Startup Name": df_new["name"],
        "Predicted Success (1=Yes, 0=No)": preds,
        "Success Probability": np.round(probs, 3)
    })
else:
    results = pd.DataFrame({
        "Predicted Success (1=Yes, 0=No)": preds,
        "Success Probability": np.round(probs, 3)
    })

print("\n✅ Prediction Results (first 10):")
print(results.head(10))

os.makedirs("outputs", exist_ok=True)
results.to_csv("outputs/predictions.csv", index=False)
print("\n💾 Saved predictions to outputs/predictions.csv")
'''
'''
import pandas as pd
import numpy as np
import joblib

# ===============================
# Load Model, Scaler, and Features
# ===============================
print("✅ Loading model, scaler, and feature names...")
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# ===============================
# Load New Data
# ===============================
data_path = "data/raw/new_startups.csv"  # update path as needed
df = pd.read_csv(data_path)
print(f"📄 Loaded new data: {df.shape}")

# ===============================
# Ensure Consistent Columns
# ===============================
# 1️⃣ Keep only numeric columns
df = df.select_dtypes(include=[np.number])

# 2️⃣ Fix potential column name typos (e.g., 'funding__log' → 'funding_log')
df.columns = [c.replace("__", "_") for c in df.columns]

# 3️⃣ Add missing columns
missing_cols = [col for col in feature_names if col not in df.columns]
if missing_cols:
    print(f"⚠️ Adding {len(missing_cols)} missing columns with 0s: {missing_cols}")
    for col in missing_cols:
        df[col] = 0

# 4️⃣ Drop any extra columns not used in training
extra_cols = [col for col in df.columns if col not in feature_names]
if extra_cols:
    print(f"⚠️ Dropping {len(extra_cols)} extra columns: {extra_cols}")
    df = df.drop(columns=extra_cols)

# 5️⃣ Reorder columns to match model training order
df = df[feature_names]

# ===============================
# Scale and Predict
# ===============================
X_scaled = scaler.transform(df)
preds = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)[:, 1]

df["predicted_success"] = preds
df["success_probability"] = probs

# ===============================
# Save Predictions
# ===============================
output_path = "data/predictions/startup_predictions.csv"
df.to_csv(output_path, index=False)
print(f"\n💾 Predictions saved to: {output_path}")
print("\n✅ Sample predictions:")
print(df[["predicted_success", "success_probability"]].head(10))
'''

import pandas as pd
import numpy as np
import joblib
import os

# ===============================
# Load Model & Scaler
# ===============================
print("✅ Loading model, scaler, and feature names...")

model = joblib.load("models/startup_success_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# ===============================
# Load New Data
# ===============================
data_path = "data/processed/engineered_startup_data.csv"
df_new = pd.read_csv(data_path)
print(f"📄 Loaded new data: {df_new.shape}")

# ===============================
# Prepare Features (Align Columns)
# ===============================
# Drop any target columns if exist
for col in ["status", "success_label"]:
    if col in df_new.columns:
        df_new = df_new.drop(columns=[col])

# Keep only numeric columns
numeric_df = df_new.select_dtypes(include=[np.number]).fillna(0)

# Align with training feature names
missing_cols = [c for c in feature_names if c not in numeric_df.columns]
extra_cols = [c for c in numeric_df.columns if c not in feature_names]

if missing_cols:
    print(f"⚠️ Adding {len(missing_cols)} missing columns with 0s: {missing_cols}")
    for c in missing_cols:
        numeric_df[c] = 0

if extra_cols:
    print(f"⚠️ Dropping {len(extra_cols)} extra columns: {extra_cols}")
    numeric_df = numeric_df.drop(columns=extra_cols)

# Reorder columns to match training
numeric_df = numeric_df[feature_names]

# ===============================
# Scale and Predict
# ===============================
X_scaled = scaler.transform(numeric_df)
preds = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)[:, 1]

# ===============================
# Results
# ===============================
if "name" in df_new.columns:
    results = pd.DataFrame({
        "Startup Name": df_new["name"],
        "Predicted Success (1=Yes, 0=No)": preds,
        "Success Probability": np.round(probs, 3)
    })
else:
    results = pd.DataFrame({
        "Predicted Success (1=Yes, 0=No)": preds,
        "Success Probability": np.round(probs, 3)
    })

print("\n✅ Prediction Results (first 10):")
print(results.head(10))

os.makedirs("outputs", exist_ok=True)
results.to_csv("outputs/predictions.csv", index=False)
print("\n💾 Saved predictions to outputs/predictions.csv")
