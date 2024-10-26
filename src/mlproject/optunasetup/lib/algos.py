from mlproject.optunasetup.lib.exceptions import UnsupportedAlgorithm
from optuna import Trial
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


def init_random_forest(trial : Trial) -> RandomForestRegressor:
  params = {
    "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
    "max_depth": trial.suggest_int("max_depth", 1, 20),
    "max_features": trial.suggest_float("max_features", 0, 1),
    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    "n_jobs": -1,
    "random_state": 42,
  }
  return RandomForestRegressor(**params)

def init_extra_forest(trial : Trial) -> ExtraTreesRegressor:
  params = {
    "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
    "max_depth": trial.suggest_int("max_depth", 1, 20),
    "max_features": trial.suggest_float("max_features", 0, 1),
    "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    "n_jobs": -1,
    "random_state": 42,
  }
  return ExtraTreesRegressor(**params)

def init_linear_regression(trial : Trial) -> LinearRegression:
  params = {
    "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
    "n_jobs": -1,
  }
  return LinearRegression(**params)

Classifier = (
  RandomForestRegressor |
  ExtraTreesRegressor |
  LinearRegression
)

def init_learner(trial : Trial) -> Classifier:
  algorithm = trial.suggest_categorical(
    "algorithm", ["linear", "forest", "extra_forest"],
  )
  if algorithm=="linear":
    model = init_linear_regression(trial)
  elif algorithm=="forest":
    model = init_random_forest(trial)
  elif algorithm=="extra_forest":
    model = init_extra_forest(trial)
  else:
    raise UnsupportedAlgorithm(f"Algorithm_name: {algorithm}")

  return model
