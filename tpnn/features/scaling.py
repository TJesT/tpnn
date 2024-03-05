from typing import Literal
import pandas as pd
import numpy as np

from ..core.pipable_base import Pipeable


def _scale_minmax(features: pd.DataFrame) -> pd.DataFrame:

    numerics = features.select_dtypes(include=np.number)
    fmin, fmax = numerics.min(), numerics.max()
    numerics -= fmin
    numerics /= fmax - fmin

    return numerics


def _scale_standart(features: pd.DataFrame) -> pd.DataFrame:

    numerics = features.select_dtypes(include=np.number)
    mean, std = numerics.mean(), numerics.std()
    numerics -= mean
    numerics /= std

    return numerics


class Scaler(Pipeable):

    __scale_mapping = {
        "minmax": _scale_minmax,
        "standart": _scale_standart,
    }

    def __init__(self, strategy: Literal["minmax", "standart"] = "minmax") -> None:
        self.scaling = self.__scale_mapping.get(strategy)

    def __call__(self, features: pd.DataFrame) -> pd.DataFrame:
        scale_results = self.scaling(features)
        return features.assign(
            **{column: scale_results[column] for column in scale_results.columns}
        )
