import base64
import json
import logging
import os
import pickle
from hashlib import sha256
from pathlib import Path

import pandas as pd
import yaml
from mlflow.models import infer_signature
from optuna import logging as optuna_logging
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn import set_config
from sklearn.metrics import mean_absolute_percentage_error

from mlproject.optunasetup.lib.exceptions import DatasetLoadingException

cur_dir = os.path.abspath(os.curdir)

logger = logging.getLogger(__name__)


def prepare_logging(log_configfile="logging.json"):
    set_config(transform_output="default")
    optuna_logging.set_verbosity(optuna_logging.WARNING)

    with open(f"conf/{log_configfile}") as f:
        config = json.load(f)

    logging.config.dictConfig(config)


def load_raw_data(dataset):
    """Loads data from input files and returns it as Pandas Dataframe.

    Args:
    ----
        config (dict): Project's configuration
        dataset (string): Which dataset needs to be loaded

    Raises:
    ------
        Exception: Given dataset was not found in the config or file was not found in local directory

    Returns:
    -------
        pd.DataFrame: DataFrame with data

    """
    data_dir = cur_dir + "/data/01_raw/" + dataset + "/"

    files = ["apartments_pl_2023_08.csv.zip"]
    for f in files:
        df = pd.read_csv(data_dir + f)
        df.name = dataset
    return df


def load_config(file="optuna-config.yml", is_test_run=False):
    """Function loads configuration from the file.

    Args:
    ----
        file (str, optional): Name of the file with configuration. Defaults to "config.yml".

    Returns:
    -------
        Config dictionary

    """
    path = "/conf/base/"
    if is_test_run:
        path = "/tests/integration_tests" + path
    conf_file = cur_dir + path + file
    with open(conf_file) as f:
        return yaml.safe_load(f)["sources"]


def load_data(dataset, is_test_run=False):
    """Loads data from input files and returns it as Pandas Dataframe.

    Args:
    ----
        config (dict): Project's configuration
        dataset (string): Which dataset needs to be loaded

    Raises:
    ------
        Exception: Given dataset was not found in the config or file was not found in local directory

    Returns:
    -------
        pd.DataFrame: DataFrame with data

    """
    path = "/data"
    if is_test_run:
        path = "/tests/integration_tests" + path
    data_dir = cur_dir + path
    filename = data_dir + "/" + dataset + ".csv"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        err_msg = f"File data/{filename} not found"
        raise DatasetLoadingException(err_msg)
    df.name = dataset
    return df


def prepare_data(config, train_split_ratio=0.2, random_stage=14, is_test_run=False):
    from sklearn.model_selection import train_test_split

    dataset = list(config.keys())[0]
    df = load_data(dataset, is_test_run)
    target = config[dataset]["target"]
    X = df.drop(target, axis=1)
    y = df[target]

    X[X.select_dtypes("object").columns.to_list()] = X[X.select_dtypes("object").columns.to_list()].astype("category")
    num_columns = list(X.select_dtypes("number").columns.values)
    cat_columns = X.select_dtypes("category").columns.to_list()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=train_split_ratio, random_state=random_stage)

    return X_train, X_val, y_train, y_val, num_columns, cat_columns, X.columns.values, target


def log_plots(study, mlflow):
    figure = None
    try:
        figure = plot_param_importances(study)
    except:
        pass
    if figure:
        mlflow.log_figure(figure, "param_importances.html")

    figure = None
    try:
        figure = plot_optimization_history(study)
    except:
        pass
    if figure:
        mlflow.log_figure(figure, "optimization_history.html")


def get_reduced_features(X_train, X_val, params, columns):
    columns_to_drop = [k for k, v in params.items() if k in columns and v == False]
    X_train_selected = X_train.drop(columns_to_drop, axis=1)
    X_val_selected = X_val.drop(columns_to_drop, axis=1)
    logger.info(f"Following columns are dropped in final model: {','.join(columns_to_drop)}")
    return X_train_selected, X_val_selected


def evaluate_model(best_model, X_train_selected, X_val_selected, y_train, y_val):
    model_bytes = base64.b64decode(best_model.encode("ascii"))
    pipeline = pickle.loads(model_bytes)
    pipeline.fit(X_train_selected, y_train)
    y_pred = pipeline.predict(X_val_selected)
    signature = infer_signature(X_train_selected, y_pred)
    validation_mape = mean_absolute_percentage_error(y_val, y_pred)
    return pipeline, signature, validation_mape


def save_best_study(study, experiment_name, X_train, y_train, X_val, y_val, columns, target, mlflow):
    """Function registers sucessfful study results to MLFlow.

    Args:
    ----
        study (Study): Optuna study object
        experiment_name (String): experiment name
        X_train (PandasDataframe): Trainig dataset for independent features
        y_train (PandasDataframe): Trainig dataset for target feature
        X_val (PandasDataframe): Validation dataset for independent features
        y_val (PandasDataframe): Validation dataset for target feature
        columns (list): list of selected columns
        target (String): target feature name
        mlflow (Mlfow): mlflow object for registering training results in MLFlow

    """
    mlflow.log_params(study.best_trial.params)
    mlflow.log_params({"target": target})
    mlflow.log_metrics({"train_mape": study.best_trial.value * (-1)})

    log_plots(study, mlflow)

    best_model = study.user_attrs["best_model"]

    mlflow.log_params({"hash": sha256(best_model.encode("utf-8")).hexdigest()[:1024]})

    X_train_selected, X_val_selected = get_reduced_features(X_train, X_val, study.best_trial.params, columns)

    pipeline, signature, validation_mape = evaluate_model(best_model, X_train_selected, X_val_selected, y_train, y_val)
    mlflow.log_metrics({"validation_mape": validation_mape})
    model_info = mlflow.sklearn.log_model(
        pipeline,
        artifact_path="ml_project",
        signature=signature,
        registered_model_name=experiment_name,
    )


def save_best_study2(study, name, X_train, y_train, X_val, y_val):
    old_model_f = Path(name + ".pkl")

    if old_model_f.is_file():
        with open(name + ".pkl", "rb") as f:
            old_pipeline = pickle.load(f)

        old_pipeline.fit(X_train, y_train)
        y_pred = old_pipeline.predict(X_val)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        print(f"Validation score of the saved model(mape): {mape}")
    else:
        print("No saved model found")

    model_bytes = bytes.fromhex(study.user_attrs["best_model"])
    pipeline = pickle.loads(model_bytes)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    print(f"Validation score(mape): {mape}")
    inpt = input("Is this model good enough? [Y/N] ")
    if inpt.lower() == "y":
        with open(name + ".pkl", "wb") as f:
            f.write(model_bytes)
