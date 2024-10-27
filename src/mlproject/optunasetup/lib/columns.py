from __future__ import annotations

from mlproject.optunasetup.lib.exceptions import InvalidColumnsSelectedException
from optuna import Trial


def choose_columns(config, all_columns: list[str]):
    dataset = list(config.keys())[0]

    exclude_columns = config[dataset].get("exclude_columns", []) or []
    include_only_columns = config[dataset].get("include_only_columns", []) or []
    parametrized_columns = config[dataset].get("parametrized_columns", []) or []

    if include_only_columns and exclude_columns:
        error_msg = "You can't use exclude and include options"
        raise InvalidColumnsSelectedException(error_msg)

    if len(set(parametrized_columns).intersection(exclude_columns)) > 0:
        error_msg = "Column name cant be in parametrized and excluded at the same time"
        raise InvalidColumnsSelectedException(error_msg)

    if len(set(parametrized_columns).intersection(include_only_columns)) > 0:
        error_msg = "Column name cant be in parametrized and included at the same time"
        raise InvalidColumnsSelectedException(error_msg)

    parametrized_columns = [col for col in parametrized_columns if col in all_columns]

    if include_only_columns:
        remain_columns = [col for col in include_only_columns if col in all_columns]
        exclude_columns = [
            col for col in all_columns if (col not in remain_columns and col not in parametrized_columns)
        ]
    else:
        remain_columns = list(set(all_columns).difference(set(parametrized_columns).union(set(exclude_columns))))

    return remain_columns, parametrized_columns, exclude_columns


def init_columns(trial: Trial, config, columns: list[str]) -> list[str]:
    remain_columns, parametrized_columns, exclude_columns = choose_columns(config, columns)

    choose = lambda column: trial.suggest_categorical(column, [True, False])
    choose_false = lambda column: trial.suggest_categorical(column, [False])
    choices = [*filter(choose, parametrized_columns)]

    choices_false = [*filter(choose_false, exclude_columns)]

    return choices + choices_false + remain_columns
