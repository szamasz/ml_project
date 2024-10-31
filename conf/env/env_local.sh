#!/bin/bash -x

export ENV_VAR=local

FULL_PATH_TO_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
DIRNAME="$(dirname "$FULL_PATH_TO_SCRIPT")"

if [[ ! -f "$DIRNAME"/secrets_local ]]; then
  echo "Error: $DIRNAME/secrets_local does not exist, quiting"
  exit 1
fi

source "$DIRNAME"/secrets_local
envsubst < "$DIRNAME"/env_docker_local_template > "$DIRNAME"/env_docker_local

export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_ARTIFACT_URI=http://localhost:5000

POSTGRES_PORT="${POSTGRES_PORT:-5432}"
export OPTUNA_DB_URI="postgresql://$OPTUNA_DB_USER:$OPTUNA_DB_PASSWORD@localhost:$POSTGRES_PORT/optuna"

echo "Loaded LOCAL ENV variables"
