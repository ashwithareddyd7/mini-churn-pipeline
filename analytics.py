import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROCESSED = Path("data/processed/telco_processed.csv")
ANALYTICS = Path("data/analytics")
ANALYTICS.mkdir(exist_ok=True)

def analytics():
    df = pd.read_csv(PROCESSED)

    df.describe(include='all').to_csv(ANALYTICS/"summary.csv")
    df['churn_flag'].value_counts().to_csv(ANALYTICS/"churn_counts.csv")

    df.groupby('Contract')['churn_flag'].mean().sort_values(ascending=False).to_csv(
        ANALYTICS/"churn_by_contract.csv"
    )

    corr = df.select_dtypes('number').corr()['churn_flag'].sort_values()
    corr.to_csv(ANALYTICS/"correlation_with_churn.csv")

    # Visualization
    plt.figure(figsize=(8,4))
    sns.countplot(x='Contract', hue='Churn', data=df)
    plt.title("Churn by Contract Type")
    plt.tight_layout()
    plt.savefig(ANALYTICS/"churn_by_contract.png")
    plt.close()

    print("Analytics created →", ANALYTICS)

if __name__ == "__main__":
    analytics()