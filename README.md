# ml-project

## Overview

EDA for the project is located in notebooks/apartments_eda.ipynb
Docs: yprofile report and presentation is located in docs/

Optuna training training script is located in src/optuna_study.py which works with dependencies in src/lib/optuna/*

Prerequisites: 
* sudo apt-get install sqlite3 
* source conf/env/local.sh

Execution: 
* ETL: mlflow run . --env-manager local -e etl_data
* mlflow run . --env-manager local -e optuna -P config_file=apartments_selected_columns.yml -P experiment_name=apartments_selected_columns_linear -P number_of_trials=100 -P sampler=Random

Options:  
  --config_file TEXT          [required]  
  --experiment_name TEXT  
  --number_of_trials INTEGER  [required]  
  --prune TEXT  
  --sampler TEXT  
  --help  

* Optuna visualisation: optuna-dashboard sqlite:///apartments.sqlite3
* MFLOW UI: mlflow server --port 5000 2>&1 &>logs/mlflow_ui.log --backend-store-uri sqlite:///ml_project-mlruns.db &
* streamlit run src/app.py -- --run_name apartments_selected_columns
