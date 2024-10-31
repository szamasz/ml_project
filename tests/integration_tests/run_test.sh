#!/bin/bash -e

FULL_PATH_TO_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
DIRNAME="$(dirname "$FULL_PATH_TO_SCRIPT")"
ROOT_DIR="$DIRNAME/../../"

cd "$ROOT_DIR"

PYTHON_BIN=.venv/bin/python
TEST_DIR="tests/integration_tests/"

export INTEGRATION_TEST=1

source conf/env/env_test.sh

docker compose -f docker/docker-compose.yml -p $ENV_VAR --profile $ENV_VAR --env-file conf/env/env_docker_$ENV_VAR up -d

echo "MLFLOW_TRACKING_URI" "$MLFLOW_TRACKING_URI"
echo "OPTUNA_DB_URI" "$OPTUNA_DB_URI"

echo "WAIT 10 seconds"
sleep 10

"$PYTHON_BIN" -m mlproject --config_file=apartments_selected_columns.yml --experiment_name=apartments_selected_columns_linear --number_of_trials=10 --sampler=Random
"$PYTHON_BIN" "$TEST_DIR"/test_model.py


docker compose -f docker/docker-compose.yml -p $ENV_VAR --profile $ENV_VAR --env-file conf/env/env_docker_$ENV_VAR down
echo "Integration test services stopped"

sudo rm -rf "$TEST_DIR"/storage-tests
echo "Deleted: $TEST_DIR/storage-tests"

echo "Integration tests FINISHED"
