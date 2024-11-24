#!/bin/bash

FULL_PATH_TO_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
DIRNAME="$(dirname "$FULL_PATH_TO_SCRIPT")"
ROOT_DIR="$DIRNAME/../../"

cd "$ROOT_DIR" || exit

source conf/env/env_local.sh

docker compose -f docker/docker-compose.yml -p $ENV_VAR --profile $ENV_VAR --env-file conf/env/env_docker_$ENV_VAR up -d

echo "MLFLOW_TRACKING_URI" "$MLFLOW_TRACKING_URI"
echo "OPTUNA_DB_URI" "$OPTUNA_DB_URI"

cd data/ || exit
trap 'echo "DVC already initialized, skipping...";' ERR

dvc init --subdir -q

DVC_REMOTE_LIST=$(dvc remote list)

if [ "$DVC_REMOTE_LIST" = "" ]; then
    echo "Setting dvc remote"
    dvc remote add -d apartments s3://data-source
    dvc remote default apartments
    dvc remote modify apartments access_key_id "${MINIO_ROOT_USER}"
    dvc remote modify apartments secret_access_key "${MINIO_ROOT_PASSWORD}"
    dvc remote modify apartments endpointurl http://localhost:9000/
    dvc remote list
else
    echo "DVC remote already configured"
fi
