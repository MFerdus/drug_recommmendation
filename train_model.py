from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "drug_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "drug_recommender.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURES = [
    "age", "sex", "bmi", "systolic_bp", "glucose", "cholesterol",
    "kidney_disease", "liver_disease", "allergy_penicillin",
    "pregnant", "condition"
]
TARGET = "recommended_drug"

def train() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    categorical = ["sex", "condition"]
    numerical = [c for c in FEATURES if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numerical),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "classes": sorted(y.unique().tolist()),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")

if __name__ == "__main__":
    train()
