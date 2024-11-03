import logging
import math
import os

import click
import mlflow
from optuna import create_study
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import RandomSampler, TPESampler

from mlproject.etl_data import process_data
from mlproject.optunasetup.lib.utils import load_config, prepare_data, save_best_study
from mlproject.optunasetup.objective import objective

run_nr = 1

logger = logging.getLogger("mlproject")


def best_model_callback(study, trial):
    global run_nr
    logger.info(f"Run nr: {run_nr}, optuna trial nr: {int(trial.number)+1}")
    run_nr += 1
    if study.best_trial.number == trial.number:
        logger.info(f"Found better model with mape: {study.best_value}")
        study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])


@click.command()
@click.option("--config_file", type=click.STRING, required=True)
@click.option("--experiment_name", type=click.STRING, required=False)
@click.option("--number_of_trials", type=click.INT, default=1, required=True)
@click.option("--prune", type=click.STRING, required=False)
@click.option("--sampler", type=click.STRING, required=False)
@click.option("--preprocess_data", is_flag=True, help="Reprocess data before training")
def main(config_file, experiment_name, number_of_trials, prune, sampler, preprocess_data):
    prune = True if prune == "True" else False
    sampler = RandomSampler() if sampler == "Random" else TPESampler()
    optuna_storage_db = os.getenv("OPTUNA_DB_URI")
    if not optuna_storage_db:
        err_msg = "Missing db storage for optuna"
        raise Exception(err_msg)
    # "postgresql://optunauser:optunapassword@localhost:5432/optuna"

    logger.debug(f"DBURI: {optuna_storage_db}")

    is_test_run = os.getenv("INTEGRATION_TEST", False) == "1"

    if preprocess_data:
        logger.info("""Reprocessing of input data""")
        process_data()

    pruner = NopPruner
    if prune:
        min_trials = math.ceil(number_of_trials / 10)
        check_trials = min_trials * 3
        check_interval = min_trials
        pruner = MedianPruner(n_startup_trials=min_trials, interval_steps=check_interval, n_min_trials=check_trials)

    if not experiment_name:
        experiment_name = config_file.split(".")[0]

    logger.info(
        f"""
    Running hyperparameter search with following options:
          * config file: {config_file}
          * experiment name: {experiment_name}
          * number of runs: {number_of_trials}
          * prune enabled: {prune}
          * sampler: {sampler!s}
    """,
    )

    config = load_config(config_file, is_test_run)

    X_train, X_val, y_train, y_val, num_columns, cat_columns, columns, target = prepare_data(
        config,
        is_test_run=is_test_run,
    )

    study = create_study(
        study_name=experiment_name,
        direction="maximize",
        pruner=pruner,
        sampler=sampler,
        storage=optuna_storage_db,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            config,
            prune,
            numerical_columns=num_columns,
            categorical_columns=cat_columns,
        ),
        callbacks=[best_model_callback],
        n_trials=number_of_trials,
    )

    logger.info(f"Best mape score on training data: {float(study.best_value)*(-1)}")

    with mlflow.start_run():
        save_best_study(study, experiment_name, X_train, y_train, X_val, y_val, columns, target, mlflow)


if __name__ == "__main__":
    # prepare_logging()
    main()
