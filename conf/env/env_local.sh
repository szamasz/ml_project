#export COMPOSE_PROFILES=local
export ENV_VAR=local

FULL_PATH_TO_SCRIPT="$(realpath "${BASH_SOURCE}")"
DIRNAME="$(dirname "$FULL_PATH_TO_SCRIPT")"

source $DIRNAME/secrets_local
envsubst < $DIRNAME/env_docker_local_template > $DIRNAME/env_docker_local

export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_ARTIFACT_URI=http://localhost:5000

POSTGRES_PORT="${POSTGRES_PORT:-5432}" 
export OPTUNA_DB_URI="postgresql://$OPTUNA_DB_USER:$OPTUNA_DB_PASSWORD@localhost:$POSTGRES_PORT/optuna"

#mlflow server --port 5000 2>&1 &>logs/mlflow_ui.log --backend-store-uri sqlite:///ml_project-mlruns.db &
#optuna-dashboard sqlite:///apartments-optuna.sqlite3 2>&1 &>logs/optuna_ui.log &