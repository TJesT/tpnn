from typing import Literal
import pandas as pd
import numpy as np

from ..core.pipeline import Pipeable


def _scale_minmax(numerics: pd.DataFrame) -> pd.DataFrame:
    fmin, fmax = numerics.min(), numerics.max()
    numerics -= fmin
    numerics /= fmax - fmin

    return numerics


def _scale_standard(numerics: pd.DataFrame) -> pd.DataFrame:
    mean, std = numerics.mean(), numerics.std()
    numerics -= mean
    numerics /= std

    return numerics


class Scaler(Pipeable[pd.DataFrame, pd.DataFrame]):

    __scale_mapping = {
        "minmax": _scale_minmax,
        "standard": _scale_standard,
    }

    def __init__(self, strategy: Literal["minmax", "standard"] = "minmax") -> None:
        self.scaling = self.__scale_mapping.get(strategy)

    def __call__(self, features: pd.DataFrame) -> pd.DataFrame:
        numerics = features.select_dtypes(include=np.number)

        scale_results = self.scaling(numerics)

        return features.assign(
            **{column: scale_results[column] for column in scale_results.columns}
        )
