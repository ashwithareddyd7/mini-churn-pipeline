
import csv
from datetime import datetime
from pathlib import Path

LOG = Path("data/prediction_logs/predictions.csv")
LOG.parent.mkdir(parents=True, exist_ok=True)

def log_prediction(features, prob):
    row = {"timestamp": datetime.utcnow().isoformat(), 
           "churn_probability": prob,
           **features}

    write_header = not LOG.exists()

    with open(LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header: 
            writer.writeheader()
        writer.writerow(row)

    print("Prediction logged →", LOG)