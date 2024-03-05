import pandas as pd
import numpy as np


def replace_categorical_values(features: pd.DataFrame) -> pd.DataFrame | None:
    def convert2codes(column: "pd.Series[pd.Categorical]") -> int:
        return column.cat.codes

    categorical = features.select_dtypes(include=object)
    categorical = categorical.astype("category").aggregate(convert2codes)

    return features, categorical
