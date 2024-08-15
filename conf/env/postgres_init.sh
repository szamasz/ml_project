if !([ "$COMPOSE_PROFILES" == "local" ] || [ "$COMPOSE_PROFILES" == "prod" ]); then
    echo "Invalid environment name: $COMPOSE_PROFILES"
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

if [ "$COMPOSE_PROFILES" == "local" ]; then
  echo "Installing optuna db"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER $OPTUNA_DB_USER with encrypted password '$OPTUNA_DB_PASSWORD';
    CREATE DATABASE optuna;
    GRANT ALL PRIVILEGES ON DATABASE optuna TO $OPTUNA_DB_USER;
EOSQL
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname optuna <<-EOSQL
        GRANT ALL ON SCHEMA public TO $OPTUNA_DB_USER;
EOSQL
fi