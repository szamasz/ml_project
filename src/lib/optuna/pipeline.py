from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from optuna import Trial
from lib.optuna.imputers import init_imputers
from lib.optuna.scalers import init_robust_scaler
from lib.optuna.encoders import init_encoder
from lib.optuna.columns import init_columns
from lib.optuna.algos import init_learner
import sys


def init_numerical_pipeline(trial : Trial, columns: list[str] = None) -> Pipeline:

  steps = [
    init_imputers(trial),
    ('scaler', init_robust_scaler(trial))
  ]
  pipeline = Pipeline(steps)

  return pipeline

def init_categorical_pipeline(trial : Trial, columns: list[str] = None) -> Pipeline:
  pipeline = Pipeline([
    ('encoder', init_encoder(trial)),
    ('scaler', init_robust_scaler(trial)) 
  ])
  return pipeline



def init_model(trial : Trial, numerical_columns : list[str], categorical_columns : list[str]) -> ColumnTransformer:

  selected_numerical_columns = init_columns(trial, numerical_columns)
  #selected_categorical_columns = init_columns(trial, categorical_columns)
  selected_categorical_columns = init_columns(trial, categorical_columns)

  numerical_pipeline = init_numerical_pipeline(trial, selected_numerical_columns)
  categorical_pipeline = init_categorical_pipeline(trial, selected_categorical_columns)
  
  #print(f"numericaL: {selected_numerical_columns}\ncategorical: {selected_categorical_columns}")

  processor = ColumnTransformer([
   ('numerical_pipeline', numerical_pipeline, selected_numerical_columns),
   ('categorical_pipeline', categorical_pipeline, selected_categorical_columns)
  ])

  learner = init_learner(trial)
  
  model = Pipeline([
    ('processor', processor),
    ('learner', learner)
  ])
  
  return model

