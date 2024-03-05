import pandas as pd
import numpy as np
from functools import partial
from typing import Any, Literal
from ..core.pipable_base import Pipeable

type column_name = str


def _find_most_frequent(series: pd.Series) -> Any:
    uniques, counts = np.unique(series.values, return_counts=True)
    return uniques[np.argmax(counts)]


def impute_most_frequent(
    features: pd.DataFrame, subset: list[column_name]
) -> pd.DataFrame:
    return features.fillna(
        {column: _find_most_frequent(features.loc[:, [column]]) for column in subset}
    )


def impute_mean(features: pd.DataFrame, subset: list[column_name]) -> pd.DataFrame:
    return features.fillna(
        {column: np.mean(features.loc[:, column].values) for column in subset}
    )


def impute_constant(
    features: pd.DataFrame, subset: list[column_name], fill_value: Any | list[Any]
) -> pd.DataFrame:
    if not isinstance(fill_value, list):
        fill_value = [fill_value] * len(subset)

    return features.fillna(
        {column: constant for column, constant in zip(subset, fill_value)}
    )


def impute_with_previous_value(
    features: pd.DataFrame, subset: list[column_name]
) -> pd.DataFrame:
    features = features.copy()
    features.loc[:, subset] = features.loc[:, subset].ffill()

    return features


def impute_with_next_value(
    features: pd.DataFrame, subset: list[column_name]
) -> pd.DataFrame:

    features = features.copy()
    features.loc[:, subset] = features.loc[:, subset].bfill()

    return features


class KeywordArgumentNotProvided(Exception):
    def __init__(self, kwarg_name: str, target_name: str, *args: object) -> None:
        super().__init__(*args)
        self.kwarg_name = kwarg_name
        self.traget_name = target_name

    def __str__(self) -> str:
        return f"Argument `{self.kwarg_name}` was not provided for `{self.traget_name}`"


class Imputer(Pipeable):
    __impute_mapping = {
        "drop": pd.DataFrame.dropna,
        "freq": impute_most_frequent,
        "mean": impute_mean,
        "const": impute_constant,
        "ffill": impute_with_previous_value,
        "bfill": impute_with_next_value,
    }

    def __init__(
        self,
        columns: Literal["all"] | list[column_name] = "all",
        strategy: Literal["drop", "freq", "mean", "const"] = "drop",
        *,
        fill_value: Any | list[Any] = None,
    ) -> None:

        self.columns = columns
        self.impute = self.__impute_mapping.get(strategy)

        if strategy == "const" and fill_value is not None:
            self.impute = partial(self.impute, fill_value=fill_value)
        else:
            raise KeywordArgumentNotProvided("fill_value", "impute_constant")

    def __call__(self, features: pd.DataFrame) -> pd.DataFrame:
        subset = self.columns

        if isinstance(subset, str):
            subset = features.columns

        return self.impute(features, subset=subset)
