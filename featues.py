import numpy as np
import pandas as pd


def remove_strongly_related_features(
    features: pd.DataFrame, thresh: float = 0.5
) -> pd.DataFrame:
    correlation_matrix = features.corr()

    # Mask to filter diagonal elements with same labels
    diagonal = np.eye(correlation_matrix.shape[0], dtype=bool)

    # Get features that have any strong relations with others
    have_any_strong_correlations = (
        correlation_matrix.mask(diagonal).abs() >= thresh
    ).any()

    # NOTE: Maybe should remove only one strongly related feature
    #       instead of all of them. Should create other func for that.
    return features[~have_any_strong_correlations]
