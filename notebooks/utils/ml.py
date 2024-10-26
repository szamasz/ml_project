from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


def fit_transform_display(model_class,X_train, y_train, X_test, y_test, model_kwargs = None):
    """Trains the given model with provided arguments and data split. Displays stats and returns the model.

    Args:
    ----
        model_class (object): ML model to be used for training and predictions
        X_train (pd.DataFrame): Training dataset for independent variables
        y_train (pd.DataFrame): Training dataset for target variable
        X_test (pd.DataFrame): Test dataset for independent variables
        y_test (pd.DataFrame): Test dataset for target variable
        model_kwargs (dict, optional): Optional arguments to the ML model. Defaults to None.

    Returns:
    -------
        object: Trained model object

    """
    if not model_kwargs:
        model_kwargs = dict()
    model_reg = model_class(**model_kwargs)
    model_reg.fit(X_train, y_train)
    y_pred = model_reg.predict(X_test)
    train_score = model_reg.score(X_train,y_train)
    test_score = model_reg.score(X_test,y_test)
    rmse=mean_squared_error(y_test,y_pred,squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"Model train score: {train_score}\nModel test score: {test_score}\nRMSE: {rmse}\nMAPE: {mape}")
    return model_reg

def best_run(cv_results_, threshold = 0.05):
    """Finds runs where not overfitting happened (train score - test score < threshould) and displays best scores and parameters

    Args:
    ----
        cv_results_ (dict): result dict from GridSearchCV or RandomizedSearchCV
        threshold (float, optional): Threshold below which train and test score is acceptable. Defaults to 0.05.

    Returns:
    -------
        dict: hyperparameters of the best run

    """
    nooverfit = [train_test  for train_test in list(zip(cv_results_["mean_train_score"], cv_results_["mean_test_score"])) if (train_test[0] - train_test[1] <= threshold)]
    if nooverfit:
        best_index = nooverfit.index(max(nooverfit))
        print(f"Number of not overfitted results {len(nooverfit)}")
        print(f"Best result:\nTrain: {cv_results_['mean_train_score'][best_index]}\nTest:{cv_results_['mean_test_score'][best_index]}")
        print(f"Best params: {cv_results_['params'][best_index]}")
        return cv_results_["params"][best_index]
    else:
        return dict()

