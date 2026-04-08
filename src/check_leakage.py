
import pandas as pd

df = pd.read_csv("data/processed/engineered_startup_data.csv")
print(df[["funding_total_usd", "funding_rounds_x", "milestones", "relationships", "status"]].corr())

print(df[["funding_total_usd", "funding_rounds_x", "relationships", "milestones", "success_label"]].describe())
print(df[["funding_total_usd", "funding_rounds_x", "relationships", "milestones", "success_label"]].corr())
