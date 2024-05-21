from functools import partial
from typing import Callable, Literal, override
import pandas as pd
import numpy as np


from ..core.exceptions import KeywordArgumentNotProvided
from ..core.types import pdPipeable, Label, Pipeable

Encoder = Callable[[pd.DataFrame, Label], pd.DataFrame]


def _encode_with_numerics(features: pd.DataFrame, subset: list[Label]) -> pd.DataFrame:
    def convert2codes(column: "pd.Series[pd.Categorical]") -> int:
        return column.cat.codes

    categorical = features[subset]
    categorical = categorical.astype("category").apply(convert2codes, axis=0)

    return categorical


def _encode_with_onehot(features: pd.DataFrame, subset: list[Label]) -> pd.DataFrame:
    codes = _encode_with_numerics(features, subset)

    def encode_column(column: "pd.Series[int]") -> "pd.Series[list[float]]":
        length = column.max() + 1

        def encode_value(value: int) -> list[float]:
            onehot = [0.0] * length
            onehot[value] = 1.0

            return onehot

        return column.apply(encode_value)

    return codes.apply(encode_column, axis=0)


def _discretize_by_bins(
    features: pd.DataFrame, subset: list[Label], bins: list[np.number]
) -> pd.DataFrame:
    discretization = features[subset].apply(
        lambda series: pd.cut(series, bins=bins, labels=False), axis=0
    )
    return discretization


def _discretize_by_quantiles(
    features: pd.DataFrame, subset: list[Label], quants: int
) -> pd.DataFrame:
    discretization = features[subset].apply(
        lambda series: pd.qcut(series, q=quants, labels=False), axis=0
    )
    return discretization


class Encoder(Pipeable[pd.DataFrame, pd.DataFrame]):
    __encode_mapping: dict[str, Encoder] = {
        "codes": _encode_with_numerics,
        "onehot": _encode_with_onehot,
        "bins": _discretize_by_bins,
        "quantiles": _discretize_by_quantiles,
    }

    def __init__(
        self,
        columns: Label | list[Label] = "all",
        strategy: Literal["codes", "onehot", "bins", "quantiles"] = "codes",
        *,
        bins: list[np.number] = None,
        quantiles: int = None,
    ):
        self.encoding = self.__encode_mapping.get(strategy)
        self.columns = columns

        if strategy == "bins":
            if bins is not None:
                self.encoding = partial(self.encoding, bins=bins)
            else:
                raise KeywordArgumentNotProvided("bins", "_discretize_by_bins")
        elif strategy == "quantiles":
            if quantiles is not None:
                self.encoding = partial(self.encoding, quants=quantiles)
            else:
                raise KeywordArgumentNotProvided(
                    "quantiles", "_discretize_by_quantiles"
                )

    @override
    def __call__(self, _input: pd.DataFrame) -> pd.DataFrame:
        columns = self.columns
        if columns == "all":
            columns = list(_input.columns)

        if not isinstance(columns, list):
            columns = [columns]

        encoding_results = self.encoding(_input, subset=columns)
        return _input.assign(
            **{column: encoding_results[column] for column in encoding_results.columns}
        )
