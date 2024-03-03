import pandas as pd
from optuna import create_study
from lib.optuna.objective import objective
from lib.utils import prepare_data, save_best_study
from sklearn import set_config
from optuna import logging as optuna_logging
import logging

logger = logging.getLogger(__name__)

def best_model_callback(study, trial):
    print(f"CALLBACK: {study.best_trial.params['algorithm']}")
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])

def main():
    set_config(transform_output="default")
    optuna_logging.set_verbosity(optuna_logging.WARNING) 

    dataset = 'apartments'

    X_train, X_val, y_train, y_val, num_columns, cat_columns, columns, target = prepare_data(dataset)   

    name = dataset
    n_trials = 1

    print(f"Starting processing for dataset: {name}")
    study = create_study(study_name=name, direction='maximize', storage="sqlite:///"+name+".sqlite3", load_if_exists  = True)
    study.optimize(lambda trial: objective(trial, X_train, y_train,numerical_columns=num_columns, categorical_columns=cat_columns), callbacks=[best_model_callback], n_trials=n_trials)
    
    print(f"Best mape score on training data: {float(study.best_value)*(-1)}")

    save_best_study(study,name,X_train, y_train, X_val, y_val, columns, target)


if __name__ == '__main__':
    main()