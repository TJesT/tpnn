import pandas as pd
import numpy as np


def remove_outliers_zscore(
    features: pd.DataFrame, zscore_thresh: int = 3
) -> pd.DataFrame:
    numerics = features.select_dtypes(include=np.number)
    means = numerics.mean(axis=0)
    stds = numerics.std(axis=0)
    exclude = (((numerics - means) / stds).abs() >= zscore_thresh).any(axis=1)
    index = exclude[exclude].index

    return features.drop(index=index)


def remove_outliers_clusterization(features: pd.DataFrame) -> pd.DataFrame:
    return features
