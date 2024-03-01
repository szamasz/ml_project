import yaml
import pandas as pd
from ydata_profiling import ProfileReport
import os

cur_dir = os.path.abspath(os.curdir)

def load_config(file = "config.yml"):
    """Function loads configuration from the file

    Args:
        file (str, optional): Name of the file with configuration. Defaults to "config.yml".

    Returns:
        Config dictionary 
    """
    with open(file, 'r') as f:
        return yaml.safe_load(f)
    

def load_data(config, dataset, full = True):
    """Loads data from input files and returns it as Pandas Dataframe

    Args:
        config (dict): Project's configuration
        dataset (string): Which dataset needs to be loaded

    Raises:
        Exception: Given dataset was not found in the config or file was not found in local directory

    Returns:
        pd.DataFrame: DataFrame with data
    """

    
    data_dir = cur_dir+'/../data/01_raw/'
    df_files = []
    try:
        files = config['sources'][dataset]['files']
    except KeyError:
        print(f"Dataset {dataset} not found in config file")
        raise Exception("Failure")
    for f in files:
        try:
            df = pd.read_csv(data_dir+f)
            df_files.append(df)
        except FileNotFoundError:
            print(f"File {data_dir}{f} not found")
            raise Exception("Failure")
    if full:
        df_res = pd.concat(df_files, axis=0, ignore_index=True)
    else:
        df_res = pd.concat(df_files[0:1], axis=0, ignore_index=True)
        #df_res = pd.concat(df_files[-1], axis=0, ignore_index=True)
    df_res.name = dataset
    return df_res

def generate_profile(df):
    cur_dir = os.path.abspath(os.curdir)
    data_dir = cur_dir+'/../docs/'
    """Generates Pandas profile to the file

    Args:
        df (pandas.DataFrame): Input DataFrame
    """
    profile = ProfileReport(df, title=f"Profiling Report for {df.name}")
    profile.to_file(data_dir + f"{df.name}.html")