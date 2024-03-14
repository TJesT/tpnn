from typing import Literal
import pandas as pd
import numpy as np

from ..core.pipeline import Pipeable
from ..core.types import pdPipeable, Label


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


class Scaler(pdPipeable):

    __scale_mapping = {
        "minmax": _scale_minmax,
        "standard": _scale_standard,
    }

    def __init__(
        self,
        columns: Literal["all"] | list[Label] = "all",
        strategy: Literal["minmax", "standard"] = "minmax",
    ) -> None:
        self.scaling = self.__scale_mapping.get(strategy)
        self.columns = columns

    def __call__(self, _input: pd.DataFrame) -> pd.DataFrame:
        if isinstance(subset, str) and subset == "all":
            subset = _input.columns

        numerics = _input[subset].select_dtypes(include=np.number)

        scale_results = self.scaling(numerics)

        return _input.assign(
            **{column: scale_results[column] for column in scale_results.columns}
        )
