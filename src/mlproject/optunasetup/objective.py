from __future__ import annotations

import base64
import logging
import pickle
from typing import Optional

import numpy as np
from optuna import Trial, TrialPruned
from pandas import DataFrame, Series
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from mlproject.optunasetup.lib.exceptions import InvalidUserInputException
from mlproject.optunasetup.pipeline import init_model

MIN_SAMPLES = 1024

logger = logging.getLogger(__name__)


def objective(
    trial: Trial,
    X: DataFrame,
    y: np.ndarray | Series,
    config,
    prune,
    numerical_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    random_state: int = 42,
) -> float:
    if not (len(numerical_columns) and len(categorical_columns)):
        err_msg = "Numerical and categorical columns must be provided!"
        raise InvalidUserInputException(err_msg)

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
            trial.report(mape_score * (-1), n_samples)
            n_samples *= 2
            n_samples = min(n_samples, all_samples)
            if trial.should_prune():
                logger.info("Run pruned!")
                raise TrialPruned()
    else:
        X_train_sample = X
        y_train_sample = y

    model = init_model(trial, config, numerical_columns, categorical_columns)

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    scores = cross_val_score(model, X_train_sample, y_train_sample, scoring=mape_scorer, cv=kf)

    trial.set_user_attr(key="best_model", value=base64.b64encode(pickle.dumps(model)).decode("ascii"))
    return np.min([np.mean(scores), np.median([scores])])
