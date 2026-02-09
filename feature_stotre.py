import pandas as pd
import sqlite3
from pathlib import Path

DB_PATH = Path("model_store/feature_store.db")
DB_PATH.parent.mkdir(exist_ok=True)

def build_feature_store():
    df = pd.read_csv("data/processed/telco_processed.csv")

    base_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'InternetService', 'Contract',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
        'tenure', 'avg_monthly_charges'
    ]

    data = df[base_cols + ['churn_flag']]
    data = pd.get_dummies(data, drop_first=True)

    conn = sqlite3.connect(DB_PATH)
    data.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()

    print("Feature store created →", DB_PATH)

if __name__ == "__main__":
    build_feature_store()