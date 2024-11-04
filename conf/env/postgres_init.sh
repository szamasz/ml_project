#!/bin/bash -x

if ! ([ "$ENV_VAR" == "local" ] || [ "$ENV_VAR" == "test" ]); then
    echo "Invalid environment name: $ENV_VAR"
    exit 1
fi

echo "Installing mlflow db"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER $MLFLOW_DB_USER with encrypted password '$MLFLOW_DB_PASSWORD';
    CREATE DATABASE mlflow;
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO $MLFLOW_DB_USER;
EOSQL
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname mlflow <<-EOSQL
    GRANT ALL ON SCHEMA public TO $MLFLOW_DB_USER;
EOSQL

echo "Installing optuna db"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER $OPTUNA_DB_USER with encrypted password '$OPTUNA_DB_PASSWORD';
    CREATE DATABASE optuna;
    GRANT ALL PRIVILEGES ON DATABASE optuna TO $OPTUNA_DB_USER;
EOSQL
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname optuna <<-EOSQL
        GRANT ALL ON SCHEMA public TO $OPTUNA_DB_USER;
EOSQL
