#!/bin/bash -x

export ENV_VAR=test

FULL_PATH_TO_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
DIRNAME="$(dirname "$FULL_PATH_TO_SCRIPT")"

source "$DIRNAME"/secrets_test

envsubst < "$DIRNAME"/env_docker_test_template > "$DIRNAME"/env_docker_test

source "$DIRNAME"/env_docker_test

export MLFLOW_TRACKING_URI=http://localhost:15000
export MLFLOW_ARTIFACT_URI=http://localhost:15000

export OPTUNA_DB_URI="postgresql://$OPTUNA_DB_USER:$OPTUNA_DB_PASSWORD@localhost:$POSTGRES_PORT/optuna"

echo "Loaded TEST ENV variables"
