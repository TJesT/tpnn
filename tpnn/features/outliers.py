from functools import partial
from typing import Literal, Optional, override
import pandas as pd
import numpy as np

from ..core.exceptions import KeywordArgumentNotProvided
from ..core.pipeline import Pipeable


def _zscore_filter(features: pd.DataFrame, zscore_thresh: int = 3) -> pd.DataFrame:
    numerics = features.select_dtypes(include=np.number)
    means = numerics.mean(axis=0)
    stds = numerics.std(axis=0)
    exclude = (((numerics - means) / stds).abs() >= zscore_thresh).any(axis=1)
    index = exclude[exclude].index

    return features.drop(index=index)


def _clusterization_filter(features: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


class OutliersFilter(Pipeable[pd.DataFrame, pd.DataFrame]):
    __filter_mapping = {
        "zscore": _zscore_filter,
        "cluster": _clusterization_filter,
    }

    def __init__(
        self,
        strategy: Literal["zscore", "cluster"] = "zscore",
        *,
        thresh: Optional[float] = None,
    ) -> None:
        self.filter = self.__filter_mapping.get(strategy)

        if strategy == "zscore" and thresh is not None:
            self.filter = partial(self.filter, thresh=thresh)

    @override
    def __call__(self, _input: pd.DataFrame) -> pd.DataFrame:
        return self.filter(_input)
