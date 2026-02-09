import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from pathlib import Path

DB = Path("model_store/feature_store.db")
MODEL_PATH = Path("model_store/churn_model.pkl")

def train():
    conn = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM features", conn)
    conn.close()

    X = df.drop(columns=['churn_flag'])
    y = df['churn_flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)

    joblib.dump(clf, MODEL_PATH)

    print(f"Model saved → {MODEL_PATH}")
    print(f"Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}")

if __name__ == "__main__":
    train()