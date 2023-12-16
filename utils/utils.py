import yaml
import pandas as pd
from ydata_profiling import ProfileReport

def load_config(file = "config.yml"):
    """Function loads configuration from the file

    Args:
        file (str, optional): Name of the file with configuration. Defaults to "config.yml".

    Returns:
        Config dictionary 
    """
    with open(file, 'r') as f:
        return yaml.safe_load(f)
    

def load_data(config, dataset):
    """Loads data from input files and returns it as Pandas Dataframe

    Args:
        config (dict): Project's configuration
        dataset (string): Which dataset needs to be loaded

    Raises:
        Exception: Given dataset was not found in the config or file was not found in local directory

    Returns:
        pd.DataFrame: DataFrame with data
    """
    df_files = []
    try:
        files = config['sources'][dataset]['files']
    except KeyError:
        print(f"Dataset {dataset} not found in config file")
        raise Exception("Failure")
    for f in files:
        try:
            df = pd.read_csv("data/"+f)
            df_files.append(df)
        except FileNotFoundError:
            print(f"File data/{f} not found")
            raise Exception("Failure")
    df_res = pd.concat(df_files, axis=0, ignore_index=True)
    df_res.name = dataset
    return df_res

def generate_profile(df):
    """Generates Pandas profile to the file

    Args:
        df (pandas.DataFrame): Input DataFrame
    """
    profile = ProfileReport(df, title=f"Profiling Report for {df.name}")
    profile.to_file(f"reports/{df.name}.html")