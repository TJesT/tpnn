import numpy as np
import pandas as pd


def calculate_entropy(feature: pd.Series) -> float:
    value_counts = feature.value_counts(normalize=True, sort=False)

    return -(value_counts * np.log2(value_counts)).sum()


# def calculate_split_information_gain(feature: pd.Series, target: pd.Series) -> float:

#     return calculate_entropy(feature > targets_std)
