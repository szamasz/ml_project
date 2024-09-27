from optuna import Trial
from sklearn.impute import KNNImputer, SimpleImputer, MissingIndicator
from sklearn.pipeline import FeatureUnion

Imputer = (
  KNNImputer |
  SimpleImputer
)

def init_knn_imputer(trial : Trial) -> Imputer:
  return  ('imputer', KNNImputer(n_neighbors=5))
  #return  KNNImputer(n_neighbors=5)

def init_simple_imputer(trial : Trial) -> Imputer:
  return ('imputer', SimpleImputer())
  #return SimpleImputer()

def init_missing_indicator(trial : Trial) -> Imputer:
  return ('imputer_missing_indicator', MissingIndicator())

def init_imputers(trial : Trial) -> Imputer:
  method = trial.suggest_categorical(
    'imputing_method', ['knn', 'simple']
  )
  missing = trial.suggest_categorical(
    'indicate_missing', ['no','yes']
  )
  
  imputers = []

  if method=='knn':
    imputer = init_knn_imputer(trial)
    imputers.append(init_knn_imputer(trial))
  elif method=='simple':
    imputer = init_simple_imputer(trial)
    imputers.append(init_simple_imputer(trial))

  if missing == 'yes':
    imputers.append(init_missing_indicator(trial))

  imputer_transformer = ('imputers', FeatureUnion(imputers))
  return imputer_transformer
  #return imputer