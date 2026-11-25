"""
Log the CURRENT price model to MLflow (experiment tracking).

What this does:
  * reads the model that is already in deployment/models/ (xgb_model.json + artifacts.json)
  * logs its settings (params), its quality numbers (metrics) and its files (artifacts)
  * every run of this script creates one new "run" you can open in the MLflow UI

What this does NOT do:
  * it does NOT train a model. The model was trained once in
    machine_learning/Airbnb_Price_Prediction.ipynb. Running this script again logs the
    same model again, so you can compare runs only after you train a different model.
  * the MAE / RMSE / R2 numbers are the ones the notebook reported on its host-disjoint
    test set. They are copied below, not recomputed here.

Run (from the deployment/ folder, with the venv on):
    pip install -r requirements-mlflow.txt
    python -m tools.log_to_mlflow
    mlflow ui --backend-store-uri sqlite:///mlflow.db
Then open http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import xgboost as xgb

DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = DEPLOYMENT_DIR / "models"

# Copied from the modeling notebook (see README "Model" table). Update these if you retrain.
NOTEBOOK_METRICS = {"test_mae_usd": 41.36, "test_rmse_usd": 60.91, "test_r2": 0.759}


def log_current_model(
    model_dir: Path = MODEL_DIR,
    tracking_uri: str = f"sqlite:///{DEPLOYMENT_DIR / 'mlflow.db'}",
    experiment: str = "travelpulse-price-model",
    run_name: str = "logged-existing-model",
) -> str:
    """Log the model in model_dir as one MLflow run. Returns the run id."""
    artifacts = json.loads((model_dir / "artifacts.json").read_text())
    model_path = model_dir / artifacts["model_file"]

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as run:
        # Tags: honest labels so nobody mistakes this for a new training run.
        mlflow.set_tags({"trained_in_this_run": "false", "source": "existing exported model"})

        # Params: settings that describe the model.
        mlflow.log_params({
            "algorithm": "XGBoost regressor on log1p(price)",
            "n_features": len(artifacts["feature_columns"]),
            "n_boosted_trees": booster.num_boosted_rounds(),
            "reference_date": artifacts.get("reference_date", "unknown"),
            "train_price_min_usd": artifacts["train_price_range"][0],
            "train_price_max_usd": artifacts["train_price_range"][1],
            "calibration_rows": artifacts.get("conformal", {}).get("n_calibration", "n/a"),
        })

        # Metrics: quality numbers (from the notebook) + the interval widths.
        mlflow.log_metrics(NOTEBOOK_METRICS)
        for level, q in artifacts.get("conformal", {}).get("quantiles", {}).items():
            mlflow.log_metric(f"conformal_quantile_{int(float(level) * 100)}", q)

        # Artifacts: the actual files, so the run can be reproduced later.
        mlflow.log_artifact(str(model_path), artifact_path="model_files")
        mlflow.log_artifact(str(model_dir / "artifacts.json"), artifact_path="model_files")
        mlflow.xgboost.log_model(booster, name="xgboost_model")

        return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--experiment", default="travelpulse-price-model")
    parser.add_argument("--run-name", default="logged-existing-model")
    args = parser.parse_args()
    run_id = log_current_model(args.model_dir, experiment=args.experiment, run_name=args.run_name)
    print(f"Logged run {run_id}")
    print("Open the UI with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
