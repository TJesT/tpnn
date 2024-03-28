import pandas as pd
import numpy as np
from functools import partial
from typing import Any, Literal

from ..core.exceptions import KeywordArgumentNotProvided
from ..core.pipeline import Pipeable
from ..core.types import Label, pdPipeable


def _find_most_frequent(series: pd.Series) -> Any:
    uniques, counts = np.unique(series.values, return_counts=True)
    return uniques[np.argmax(counts)]


def _impute_most_frequent(features: pd.DataFrame, subset: list[Label]) -> pd.DataFrame:
    return features.fillna(
        {column: _find_most_frequent(features.loc[:, [column]]) for column in subset}
    )


def _impute_mean(features: pd.DataFrame, subset: list[Label]) -> pd.DataFrame:
    return features.fillna(
        {column: np.mean(features.loc[:, column].values) for column in subset}
    )


def _impute_constant(
    features: pd.DataFrame, subset: list[Label], fill_value: Any | list[Any]
) -> pd.DataFrame:
    if not isinstance(fill_value, list):
        fill_value = [fill_value] * len(subset)

    return features.fillna(
        {column: constant for column, constant in zip(subset, fill_value)}
    )


def _impute_with_previous_value(
    features: pd.DataFrame, subset: list[Label]
) -> pd.DataFrame:
    features = features.copy()
    features.loc[:, subset] = features.loc[:, subset].ffill()

    return features


def _impute_with_next_value(
    features: pd.DataFrame, subset: list[Label]
) -> pd.DataFrame:

    features = features.copy()
    features.loc[:, subset] = features.loc[:, subset].bfill()

    return features


class Imputer(pdPipeable):
    __impute_mapping = {
        "drop": pd.DataFrame.dropna,
        "freq": _impute_most_frequent,
        "mean": _impute_mean,
        "const": _impute_constant,
        "ffill": _impute_with_previous_value,
        "bfill": _impute_with_next_value,
    }

    def __init__(
        self,
        columns: Literal["all"] | list[Label] = "all",
        strategy: Literal["drop", "freq", "mean", "const", "ffill", "bfill"] = "drop",
        /
        fill_value: Any | list[Any] = None,
    ) -> None:

        self.columns = columns
        self.impute = self.__impute_mapping.get(strategy)

        if strategy == "const":
            if fill_value is not None:
                self.impute = partial(self.impute, fill_value=fill_value)
            else:
                raise KeywordArgumentNotProvided("fill_value", "impute_constant")

    def __call__(self, _input: pd.DataFrame) -> pd.DataFrame:
        subset = self.columns

        if isinstance(subset, str) and subset == "all":
            subset = _input.columns

        return self.impute(_input, subset=self.columns)
