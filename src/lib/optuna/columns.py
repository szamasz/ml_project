from optuna import Trial
from lib.utils import load_config

dataset = 'apartments'

def choose_columns(all_columns : list[str]):
    
    config = load_config()
    #print(f'config {config}')
    exclude_columns = config['sources'][dataset].get('exclude_columns',[]) or []
    include_only_columns = config['sources'][dataset].get('include_only_columns',[]) or []
    parametrized_columns = config['sources'][dataset].get('parametrized_columns',[]) or []

    #print(f">>>>>>>>>>>>>>>all: {all_columns}")
    if include_only_columns and exclude_columns:
        raise Exception("You can't use exclude and include options")

    if len(set(parametrized_columns).intersection(exclude_columns))>0:
        raise Exception("Column name cant be in parametrized and excluded at the same time")
    
    if len(set(parametrized_columns).intersection(include_only_columns))>0:
        raise Exception("Column name cant be in parametrized and included at the same time")
    
    parametrized_columns = [col for col in parametrized_columns if col in all_columns]

    if include_only_columns:
        remain_columns = [col for col in include_only_columns if col in all_columns]
        exclude_columns = [col for col in all_columns if (col not in remain_columns and col not in parametrized_columns)]
    else:
        remain_columns = list(set(all_columns).difference(set(parametrized_columns).union(set(exclude_columns))))
    
    return remain_columns, parametrized_columns, exclude_columns

def init_columns(trial : Trial, columns : list[str]) -> list[str]:

  remain_columns, parametrized_columns, exclude_columns = choose_columns(columns)
  choose = lambda column: trial.suggest_categorical(column, [True, False])
  choose_false = lambda column: trial.suggest_categorical(column, [False])
  choices = [*filter(choose, parametrized_columns)]
  #choices_true =  [*filter(choose_true, include_columns)]
  choices_false =  [*filter(choose_false, exclude_columns)]
  #print(f"choices: {choices}")
  return choices + choices_false + remain_columns