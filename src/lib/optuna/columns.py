from optuna import Trial
from lib.utils import load_config
import sys

dataset = 'apartments'

def choose_columns(all_columns : list[str]):
    
    config = load_config()
    #print(f'config {config}')
    exclude_columns = config['sources'][dataset].get('exclude_columns',[]) or []
    include_only_columns = config['sources'][dataset].get('include_only_columns',[]) or []
    parametrized_columns = config['sources'][dataset].get('parametrized_columns',[]) or []

    print(f"HERE: {exclude_columns},{include_only_columns},{parametrized_columns}, {all_columns}")

    if include_only_columns and exclude_columns:
        raise Exception("You can't use exclude and include options")

    if len(set(parametrized_columns).intersection(exclude_columns))>0:
        raise Exception("Column name cant be in parametrized and excluded at the same time")
    
    if len(set(parametrized_columns).intersection(include_only_columns))>0:
        raise Exception("Column name cant be in parametrized and included at the same time")
    
    if diff := set(parametrized_columns).difference(all_columns):
        raise Exception(f"Provided parameterized columns are not present in the dataset: {diff}")

    if diff := set(include_only_columns).difference(all_columns):
        raise Exception(f"Provided included columns are not present in the dataset: {diff}")

    if diff := set(exclude_columns).difference(all_columns):
        raise Exception(f"Provided exlusions columns are not present in the dataset: {diff}")

    if include_only_columns:
        selected_columns = include_only_columns
    else:
        selected_columns = all_columns
        
    remain_columns = list(set(selected_columns).difference(set(exclude_columns)))
    print(f"remain: {remain_columns}, paramterized: {parametrized_columns}")
    sys.exit()
    return remain_columns, parametrized_columns

def init_columns(trial : Trial, columns : list[str]) -> list[str]:

  remain_columns, parametrized_columns = choose_columns(columns)
  #print(f"rmain: {remain_columns}")

  choose = lambda column: trial.suggest_categorical(column, [True, False])
  choices = [*filter(choose, parametrized_columns)]
  #choices_true =  [*filter(choose_true, include_columns)]
  #choices_false =  [*filter(choose_false, exclude_columns)]
  #print(f"choices: {choices}")
  return choices + remain_columns