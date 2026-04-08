# src/feature_engineering.py

import pandas as pd
import os

# === Paths ===
input_path = os.path.join("data", "processed", "merged_startup_data.csv")
output_path = os.path.join("data", "processed", "engineered_startup_data.csv")

# === Load merged data ===
df = pd.read_csv(input_path)
print("✅ Loaded merged data:", df.shape)

# === Drop irrelevant or mostly empty columns ===
drop_cols = [c for c in df.columns if df[c].nunique() <= 1 or df[c].isna().sum() > 0.8 * len(df)]
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# === Fill missing values ===
df = df.fillna(0)

# === Convert date columns to numeric (if any) ===
for col in df.columns:
    if "date" in col or "at" in col:
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].apply(lambda x: x.year if pd.notnull(x) else 0)
        except Exception:
            pass

# === Generate additional numeric ratios ===
if "funding_total_usd" in df.columns and "funding_rounds" in df.columns:
    df["avg_funding_per_round"] = df["funding_total_usd"] / (df["funding_rounds"] + 1)

if "funding_total_usd" in df.columns and "milestones" in df.columns:
    df["funding_per_milestone"] = df["funding_total_usd"] / (df["milestones"] + 1)

if "funding_rounds" in df.columns and "relationships" in df.columns:
    df["relationship_to_round_ratio"] = df["relationships"] / (df["funding_rounds"] + 1)

# === Define target variable ===
if "status" in df.columns:
    df["success_label"] = df["status"].apply(lambda x: 1 if str(x).lower() in ["acquired", "ipo", "successful"] else 0)
elif "success_score" in df.columns:
    df["success_label"] = (df["success_score"] > df["success_score"].median()).astype(int)
else:
    # fallback heuristic
    df["success_label"] = (df["funding_total_usd"] > df["funding_total_usd"].median()).astype(int)

# === Save engineered data ===
os.makedirs("data/processed", exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Feature engineering complete! Saved to: {output_path}")
print("📊 Final dataset shape:", df.shape)
