from optuna import Trial
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

Encoder = OrdinalEncoder | OneHotEncoder


def init_ohe_encoder(trial: Trial) -> OneHotEncoder:
    return OneHotEncoder(sparse_output=False, handle_unknown="error")


def init_ordinal_encoder(trial: Trial) -> OrdinalEncoder:
    return OrdinalEncoder(handle_unknown="error")


def init_encoder(trial: Trial) -> Encoder:
    method = trial.suggest_categorical(
        "encoding_method",
        ["ordinal", "onehot"],
    )

    if method == "ordinal":
        encoder = init_ordinal_encoder(trial)
    elif method == "onehot":
        encoder = init_ohe_encoder(trial)

    return encoder
