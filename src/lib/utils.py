import yaml
import pandas as pd
import os
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_percentage_error
import mlflow
from mlflow.models import infer_signature
from hashlib import sha256
from optuna.visualization import plot_param_importances
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.tools import mpl_to_plotly

cur_dir = os.path.abspath(os.curdir)

def load_raw_data(dataset):
    """Loads data from input files and returns it as Pandas Dataframe

    Args:
        config (dict): Project's configuration
        dataset (string): Which dataset needs to be loaded

    Raises:
        Exception: Given dataset was not found in the config or file was not found in local directory

    Returns:
        pd.DataFrame: DataFrame with data
    """

    
    data_dir = cur_dir+'/data/01_raw/'+dataset+'/'

    files = ['apartments_pl_2023_08.csv.zip']
    for f in files:
        df = pd.read_csv(data_dir+f)
        df.name = dataset
    return df

def load_config(file = "optuna-config.yml"):
    """Function loads configuration from the file

    Args:
        file (str, optional): Name of the file with configuration. Defaults to "config.yml".

    Returns:
        Config dictionary 
    """
    conf_file = cur_dir + '/conf/base/' + file 
    with open(conf_file, 'r') as f:
        return yaml.safe_load(f)
    

def load_data(dataset):
    """Loads data from input files and returns it as Pandas Dataframe

    Args:
        config (dict): Project's configuration
        dataset (string): Which dataset needs to be loaded

    Raises:
        Exception: Given dataset was not found in the config or file was not found in local directory

    Returns:
        pd.DataFrame: DataFrame with data
    """

    data_dir = cur_dir+'/data/05_model_input'
    filename = data_dir+'/'+dataset+".csv"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"File data/{filename} not found")
        raise Exception("Failure")
    df.name = dataset
    return df

def prepare_data(dataset, train_split_ratio = 0.2, random_stage = 14):
    from sklearn.model_selection import train_test_split

    config = load_config()
    df = load_data(dataset)
    target = config['sources'][dataset]['target']
    X = df.drop(target, axis =1)
    y = df[target]

    X[X.select_dtypes("object").columns.to_list()]  = X[X.select_dtypes("object").columns.to_list()].astype('category')
    num_columns = list(X.select_dtypes("number").columns.values)
    cat_columns = X.select_dtypes("category").columns.to_list()
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=train_split_ratio, random_state=random_stage)
    
    return  X_train, X_val, y_train, y_val, num_columns, cat_columns, X.columns.values, target

def save_best_study(study, name, X_train, y_train, X_val, y_val, columns, target):
    #with mlflow.start_run(run_name=str(study.best_trial.number)):
    with mlflow.start_run(run_name="psz_run"):
        mlflow.log_params(study.best_trial.params)
        mlflow.log_params({"target": target} )
        mlflow.log_metrics({"train_mape": study.best_trial.value*(-1)})

        figure = None
        try:
            figure = plot_param_importances(study)
        except:
            pass
        if figure:
            mlflow.log_figure(figure,'param_importances.html')

        
        columns_to_drop = [ k for k,v in study.best_trial.params.items() if k in columns and v == False ]
        print(f"Following columns are dropped in final model: {','.join(columns_to_drop)}")
        best_model = study.user_attrs['best_model']
        mlflow.log_params({"hash": sha256(best_model.encode('utf-8')).hexdigest()} )
        model_bytes = bytes.fromhex(best_model)
        pipeline = pickle.loads(model_bytes)
    
        X_train_selected = X_train.drop(columns_to_drop, axis = 1)
        X_val_selected = X_val.drop(columns_to_drop, axis = 1)
        pipeline.fit(X_train_selected,y_train)
        y_pred = pipeline.predict(X_val_selected)

        signature = infer_signature(X_train_selected, y_pred)
        validation_mape = mean_absolute_percentage_error(y_val, y_pred)
        mlflow.log_metrics({"validation_mape": validation_mape})
        mlflow.sklearn.log_model(pipeline,artifact_path="ml_project", signature=signature)

def save_best_study2(study, name, X_train, y_train, X_val, y_val):

    old_model_f = Path(name + '.pkl')

    if old_model_f.is_file():
        with open(name + '.pkl','rb') as f:
            old_pipeline = pickle.load(f)

        old_pipeline.fit(X_train,y_train)
        y_pred = old_pipeline.predict(X_val)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        print(f"Validation score of the saved model(mape): {mape}")
    else:
        print("No saved model found")

    model_bytes = bytes.fromhex(study.user_attrs['best_model'])
    pipeline = pickle.loads(model_bytes)
    
    pipeline.fit(X_train,y_train)
    y_pred = pipeline.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, y_pred)
    print(f"Validation score(mape): {mape}")
    inpt = input("Is this model good enough? [Y/N] ")
    if inpt.lower() == "y":
        with open(name + '.pkl','wb') as f:
            f.write(model_bytes)
