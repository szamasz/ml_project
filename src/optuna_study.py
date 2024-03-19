import pandas as pd
from optuna import create_study
from lib.optuna.objective import objective
from lib.utils import prepare_data, save_best_study
from sklearn import set_config
from optuna import logging as optuna_logging
from optuna.pruners import MedianPruner, NopPruner
import logging
from lib.utils import load_config
import math
import click
from optuna.samplers import RandomSampler,TPESampler


logger = logging.getLogger(__name__)

run_nr = 1

def best_model_callback(study, trial):
    global run_nr
    print(f"Run nr: {run_nr}, optuna trial nr: {int(trial.number)+1}")
    run_nr += 1
    if study.best_trial.number == trial.number:
        print(f"Found better model with mape: {study.best_value}")
        study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])

@click.command()
@click.option('--config_file', type=click.STRING, required=True)
@click.option('--experiment_name', type=click.STRING, required=False)
@click.option('--number_of_trials', type=click.INT, default=1, required=True)
@click.option("--prune", type=click.STRING, required=False)
@click.option("--sampler", type=click.STRING, required=False)
def main(config_file,experiment_name, number_of_trials, prune, sampler):
    set_config(transform_output="default")
    optuna_logging.set_verbosity(optuna_logging.WARNING) 
    prune = True if prune == "True" else False
    sampler = RandomSampler() if sampler == "Random" else TPESampler()

    pruner = NopPruner
    if prune:
        min_trials = math.ceil(number_of_trials/10)
        check_trials = min_trials * 3
        check_interval = min_trials
        pruner = MedianPruner(n_startup_trials=min_trials,interval_steps=check_interval,n_min_trials=check_trials)


    if not experiment_name:
        experiment_name = config_file.split('.')[0]

    print(f"""
    Running hyperparameter search with following options:
          * config file: {config_file}
          * experiment name: {experiment_name}
          * number of runs: {number_of_trials}
          * prune enabled: {prune}
          * sampler: {str(sampler)}
    """)

    config = load_config(config_file)



    X_train, X_val, y_train, y_val, num_columns, cat_columns, columns, target = prepare_data(config)

    study = create_study(study_name=experiment_name, direction='maximize', pruner=pruner, sampler=sampler, storage="sqlite:///mlflow_runs.sqlite3", load_if_exists  = True)
    study.optimize(lambda trial: objective(trial, X_train, y_train,config,prune, numerical_columns=num_columns, categorical_columns=cat_columns), callbacks=[best_model_callback], n_trials=number_of_trials)
    
    print(f"Best mape score on training data: {float(study.best_value)*(-1)}")

    save_best_study(study,experiment_name,X_train, y_train, X_val, y_val, columns, target)


if __name__ == '__main__':
    main()