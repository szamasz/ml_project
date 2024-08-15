export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_ARTIFACT_URI=http://localhost:5000


#mlflow server --port 5000 2>&1 &>logs/mlflow_ui.log --backend-store-uri sqlite:///ml_project-mlruns.db &
#optuna-dashboard sqlite:///apartments-optuna.sqlite3 2>&1 &>logs/optuna_ui.log &