from optuna import Trial
from typing import Optional
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from pandas import DataFrame, Series
import numpy as np
import pickle
from lib.optuna.pipeline import init_model


def objective(trial : Trial, X : DataFrame, y : np.ndarray | Series, numerical_columns : Optional[list[str]]=None, categorical_columns : Optional[list[str]]=None, random_state : int=42) -> float:
  
  if not (len(numerical_columns) and len(categorical_columns)):
    raise Exception("Numerical and categorical columns must be provided!")
  n_samples = X.shape[0]
  X_train_sample = X.sample(n_samples, random_state=random_state)
  y_train_sample = y.sample(n_samples, random_state=random_state)
  #print(categorical_columns, X.columns)
  model = init_model(trial, numerical_columns, categorical_columns)
  #print(trial.user_attrs)
  kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
  mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
  scores = cross_val_score(model, X_train_sample, y_train_sample, scoring=mape_scorer, cv=kf)
  #model_to_save = model['learner']
  trial.set_user_attr(key="best_model", value=pickle.dumps(model).hex())
  return np.min([np.mean(scores), np.median([scores])])