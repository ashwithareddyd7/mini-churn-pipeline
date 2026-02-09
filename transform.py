import pandas as pd
from pathlib import Path

raw_files = list(Path("raw_zone").glob("*.csv"))
if not raw_files:
    raise FileNotFoundError("No CSV found in raw_zone")

INGEST = raw_files[0]

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def transform():
    df = pd.read_csv(INGEST)
    df.columns = df.columns.str.strip()

    # Clean numeric values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(
        df['MonthlyCharges'] * df['tenure']
    )

    # Derived features
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=['0-6','7-12','13-24','25-48','49-72']
    )

    df['avg_monthly_charges'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
    df['churn_flag'] = df['Churn'].map({'Yes': 1, 'No': 0})

    df.to_csv(OUT_DIR / "telco_processed.csv", index=False)
    print("Processed data saved →", OUT_DIR / "telco_processed.csv")

if __name__ == "__main__":
    transform()