from typing import Literal, override
import pandas as pd
import numpy as np

from ..core.pipeline import Pipeable


def _encode_with_numerics(features: pd.DataFrame) -> pd.DataFrame:
    def convert2codes(column: "pd.Series[pd.Categorical]") -> int:
        return column.cat.codes

    categorical = features.select_dtypes(include=object, exclude=np.number)
    categorical = categorical.astype("category").aggregate(convert2codes)

    return categorical


def _encode_with_onehot(
    features: pd.DataFrame, new_columns: bool = False
) -> pd.DataFrame:
    raise NotImplementedError


class Encoder(Pipeable[pd.DataFrame, pd.DataFrame]):
    __encode_mapping = {
        "codes": _encode_with_numerics,
        "onehot": _encode_with_onehot,
    }

    def __init__(self, strategy: Literal["codes", "onehot"] = "codes"):
        self.encoding = self.__encode_mapping.get(strategy)

    @override
    def __call__(self, _input: pd.DataFrame) -> pd.DataFrame:
        return self.encoding(_input)
