"""
Trains XGBoost regressor to predict 14-day risk_pct from the environmental
vector, then fits a SHAP TreeExplainer on top for the Analytical Node's
top_3_environmental_stressors output.

Run: python3 train_model.py
Produces: model.json, explainer.pkl (in this directory)
"""

import json
import pickle

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "temperature_c", "humidity_pct", "rainfall_mm_14d",
    "soil_n", "soil_p", "soil_k", "soil_moisture_pct",
]

MODEL_PATH = "model.json"
EXPLAINER_PATH = "explainer.pkl"


def train():
    df = pd.read_csv("synthetic_environmental_risk.csv")
    X = df[FEATURE_COLUMNS]
    y = df["risk_pct"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE: {mae:.2f} percentage points")
    print(f"Test R^2: {r2:.4f}")

    # Feature importance sanity check (should roughly track heuristic weights:
    # humidity/rainfall should dominate, K/P deficiency should matter least)
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    print("\nFeature importances:")
    for feat, imp in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {feat:20s} {imp:.4f}")

    model.save_model(MODEL_PATH)

    explainer = shap.TreeExplainer(model)
    with open(EXPLAINER_PATH, "wb") as f:
        pickle.dump(explainer, f)

    with open("training_metrics.json", "w") as f:
        json.dump({"mae": mae, "r2": r2, "feature_importances": importances, "n_train": len(X_train), "n_test": len(X_test)}, f, indent=2)

    print(f"\nSaved {MODEL_PATH} and {EXPLAINER_PATH}")
    return model, explainer


if __name__ == "__main__":
    train()
