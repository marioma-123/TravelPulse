"""The MLflow logging script writes a run with params, metrics and the model files."""
import pytest

mlflow = pytest.importorskip("mlflow")  # skipped if mlflow is not installed

from tools.log_to_mlflow import NOTEBOOK_METRICS, log_current_model  # noqa: E402


def test_log_current_model(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_id = log_current_model(tracking_uri=uri, experiment="test-exp")
    run = mlflow.tracking.MlflowClient(uri).get_run(run_id)
    assert run.data.tags["trained_in_this_run"] == "false"
    assert run.data.params["n_features"] == "59"
    assert run.data.metrics["test_mae_usd"] == NOTEBOOK_METRICS["test_mae_usd"]
    assert "conformal_quantile_80" in run.data.metrics
