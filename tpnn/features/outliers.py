from functools import partial
from typing import Literal, Optional, override
import pandas as pd
import numpy as np

from tpnn.core.exceptions import KeywordArgumentNotProvided

from ..core.pipeline import Pipeable
from ..core.types import Label, pdPipeable


def _zscore_filter(features: pd.DataFrame, thresh: int = 3) -> pd.DataFrame:
    numerics = features.select_dtypes(include=np.number)
    means = numerics.mean(axis=0)
    stds = numerics.std(axis=0)
    exclude = (((numerics - means) / stds).abs() >= thresh).any(axis=1)
    index = exclude[exclude].index

    return features.drop(index=index)


def _clusterization_filter(features: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


class OutliersFilter(pdPipeable):
    __filter_mapping = {
        "zscore": _zscore_filter,
        "cluster": _clusterization_filter,
    }

    def __init__(
        self,
        column: Label,
        strategy: Literal["zscore", "cluster"] = "zscore",
        *,
        thresh: Optional[float] = None,
    ) -> None:
        self.column = column
        self.filter = self.__filter_mapping.get(strategy)

        if strategy == "zscore":
            if thresh is not None:
                self.filter = partial(self.filter, thresh=thresh)
            else:
                raise KeywordArgumentNotProvided("thresh", "_zscore_filter")

    @override
    def __call__(self, _input: pd.DataFrame) -> pd.DataFrame:
        return self.filter(_input[self.column])
