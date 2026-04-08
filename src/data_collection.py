import pandas as pd
import os

# === Paths ===
base_raw = os.path.join("data", "raw")
processed_path = os.path.join("data", "processed", "merged_startup_data.csv")

# === Load Datasets ===
success_df = pd.read_csv(os.path.join(base_raw, "startup_success_kaggle.csv"))
growth_df = pd.read_csv(os.path.join(base_raw, "startup_growth_and_funding_trends.csv"))
global_df = pd.read_csv(os.path.join(base_raw, "global_startup_success_dataset.csv"))

print("✅ Loaded datasets:")
print("Startup success:", success_df.shape)
print("Growth trends:", growth_df.shape)
print("Global startup:", global_df.shape)

# === Clean and Normalize Column Names ===
for df in [success_df, growth_df, global_df]:
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# === Remove unwanted 'unnamed' columns ===
for df in [success_df, growth_df, global_df]:
    df.drop(columns=[c for c in df.columns if "unnamed" in c], inplace=True, errors='ignore')

# === Function to Detect Startup Name Column ===
def find_name_column(df, df_name):
    name_cols = [c for c in df.columns if any(k in c for k in ["startup", "name", "company", "organization"])]
    if name_cols:
        print(f"🔍 Detected name column in {df_name}: '{name_cols[0]}'")
        return name_cols[0]
    else:
        print(f"⚠️ No clear name column found in {df_name}")
        print(f"Columns available: {df.columns.tolist()[:10]}")
        return None

success_key = find_name_column(success_df, "startup_success_kaggle")
growth_key = find_name_column(growth_df, "startup_growth_and_funding_trends")
global_key = find_name_column(global_df, "global_startup_success_dataset")

# === Merge Logic ===
merged = success_df.copy()

if success_key and growth_key:
    merged = pd.merge(merged, growth_df, left_on=success_key, right_on=growth_key, how="left")
else:
    print("⚠️ Skipping growth merge (missing key)")

if success_key and global_key:
    merged = pd.merge(merged, global_df, left_on=success_key, right_on=global_key, how="left")
else:
    print("⚠️ Skipping global merge (missing key)")

print(f"✅ Merged dataset shape: {merged.shape}")

# === Clean and Feature Enrichment ===
merged = merged.drop_duplicates(subset=[success_key])
merged = merged.fillna(0)

# Create enriched financial metrics if available
if "total_funding_usd" in merged.columns and "valuation_$b" in merged.columns:
    merged["funding_to_valuation_ratio"] = merged["total_funding_usd"] / (merged["valuation_$b"] * 1e9 + 1)
if "annual_revenue_$m" in merged.columns and "number_of_employees" in merged.columns:
    merged["revenue_per_employee"] = merged["annual_revenue_$m"] / (merged["number_of_employees"] + 1)
if "funding_rounds" in merged.columns and "age_first_funding_year" in merged.columns:
    merged["funding_intensity"] = merged["funding_rounds"] / (merged["age_first_funding_year"] + 1)

os.makedirs("data/processed", exist_ok=True)
merged.to_csv(processed_path, index=False)

print(f"\n✅ Data collection and merging complete! Saved to: {processed_path}")
print("📊 Final dataset shape:", merged.shape)
