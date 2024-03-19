from optuna import Trial
from typing import Optional
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from pandas import DataFrame, Series
import numpy as np
import pickle
from lib.optuna.pipeline import init_model
from sklearn.model_selection import train_test_split
from optuna import TrialPruned
import base64


MIN_SAMPLES = 1024

def objective(trial : Trial, X : DataFrame, y : np.ndarray | Series, config, prune, numerical_columns : Optional[list[str]]=None, categorical_columns : Optional[list[str]]=None, random_state : int=42) -> float:
  
  if not (len(numerical_columns) and len(categorical_columns)):
    raise Exception("Numerical and categorical columns must be provided!")
  
  model = init_model(trial, config, numerical_columns, categorical_columns)
  if prune:
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=random_state)
    X_train = X_train.drop_duplicates()
    
    n_samples = MIN_SAMPLES
    all_samples = X_train.shape[0]
    while n_samples < all_samples:
      X_train_sample = X_train.sample(n_samples, replace=False, random_state=random_state)
      y_train_sample = y_train[X_train_sample.index]
      model.fit(X_train_sample, y_train_sample.values.ravel())
      mape_score = mean_absolute_percentage_error(y_test, model.predict(X_test))
      trial.report(mape_score*(-1), n_samples)
      n_samples *= 2
      if n_samples > all_samples:
        n_samples = all_samples
      if trial.should_prune():
        print("Run pruned!")
        raise TrialPruned()
  else:
    X_train_sample = X
    y_train_sample = y
  #print(categorical_columns, X.columns)

  model = init_model(trial, config, numerical_columns, categorical_columns)
  #print(trial.user_attrs)
  kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
  mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
  scores = cross_val_score(model, X_train_sample, y_train_sample, scoring=mape_scorer, cv=kf)
  #model_to_save = model['learner']
  trial.set_user_attr(key="best_model", value=base64.b64encode(pickle.dumps(model)).decode('ascii'))
  return np.min([np.mean(scores), np.median([scores])])