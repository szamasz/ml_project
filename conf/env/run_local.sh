#!/bin/bash -e

FULL_PATH_TO_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
DIRNAME="$(dirname "$FULL_PATH_TO_SCRIPT")"
ROOT_DIR="$DIRNAME/../../"

cd "$ROOT_DIR"

source conf/env/env_local.sh

docker compose -f docker/docker-compose.yml -p $ENV_VAR --profile $ENV_VAR --env-file conf/env/env_docker_$ENV_VAR up -d

echo "MLFLOW_TRACKING_URI" "$MLFLOW_TRACKING_URI"
echo "OPTUNA_DB_URI" "$OPTUNA_DB_URI"
