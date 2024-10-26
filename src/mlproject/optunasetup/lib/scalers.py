from optuna import Trial
from sklearn.preprocessing import RobustScaler


# RobustScaler better than StandardScaler because of outliers
def init_robust_scaler(trial: Trial) -> RobustScaler:
    params = {
        "with_centering": trial.suggest_categorical(
            "with_centering",
            [True, False],
        ),
        "with_scaling": trial.suggest_categorical(
            "with_scaling",
            [True, False],
        ),
    }
    return RobustScaler(**params)
