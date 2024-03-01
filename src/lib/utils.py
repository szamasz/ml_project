import yaml
import pandas as pd
import os
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_percentage_error

cur_dir = os.path.abspath(os.curdir)

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
    
    return  X_train, X_val, y_train, y_val, num_columns, cat_columns

def save_best_study(study, name, X_train, y_train, X_val, y_val):

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
            print(f"L: {len(model_bytes)}")
            f.write(model_bytes)
