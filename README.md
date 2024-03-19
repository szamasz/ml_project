# ml-project

## Overview

EDA for the project is located in notebooks/apartments_eda.ipynb

Optuna training training script is located in src/optuna_study.py which works with dependencies in src/lib/optuna/*

Prerequisites: installed sqlite db

Execution: optuna-dashboard sqlite:///apartments.sqlite3

Visualisation: optuna-dashboard sqlite:///apartments.sqlite3

sudo apt-get install sqlite3

mlflow run . --env-manager local -e optuna -P config_file=apartments_selected_columns_geo_param.yml -P experiment_name=apartments_selected_columns_geo_param_sampler -P number_of_trials=100 -P sampler=Random

mlflow run . --env-manager local -e etl_data

streamlit run src/app.py -- --run_name apartments_selected_columns