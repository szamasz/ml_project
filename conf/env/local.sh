export MLFLOW_TRACKING_URI=sqlite:///ml_project-mlruns.db
export MLFLOW_ARTIFACT_URI=sqlite:///ml_project-mlruns.db


mlflow server --port 5000 2>&1 &>logs/mlflow_ui.log --backend-store-uri sqlite:///ml_project-mlruns.db &