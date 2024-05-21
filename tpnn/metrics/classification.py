import numpy as np
import pandas as pd

from ..core.types import Label, Probabilty


def entropy(feature: pd.Series) -> float:
    value_probabilities = feature.value_counts(normalize=True, sort=False)

    return -(value_probabilities * np.log2(value_probabilities)).sum()


def information_gain(dataframe: pd.DataFrame, feature: Label, target: Label) -> float:
    target_entropy = entropy(dataframe[target])

    if feature == target:
        return target_entropy

    dataframe = dataframe[[feature, target]]

    def target_probabilty(subset: pd.DataFrame) -> float:
        return subset.size / dataframe.shape[0]

    def subset_weighted_entropy(subset: pd.DataFrame) -> float:
        return target_probabilty(subset) * entropy(subset[target])

    weighted_feature_entropy = (
        dataframe.groupby(dataframe[feature])
        .apply(subset_weighted_entropy, include_groups=False)
        .sum()
    )

    return target_entropy - weighted_feature_entropy


def gain_ratio(dataframe: pd.DataFrame, feature: str, target: str) -> float:
    split_info = entropy(dataframe[feature])
    if split_info == 0:
        return 0.0

    info_gain = information_gain(dataframe, feature, target)

    return info_gain / split_info


def gain_ratios(dataframe: pd.DataFrame, target: str) -> list[tuple[Label, float]]:
    # NOTE: `enumerate` and `i` here only for sorting purposes
    ratios = [
        (gain_ratio(dataframe, column, target), i, column)
        for i, column in enumerate(dataframe.columns)
    ]

    return [(column, ratio) for ratio, _, column in sorted(ratios, reverse=True)]


def precision(
    prediction: np.ndarray[float], target: np.ndarray[float], strategy="weighted"
) -> float:
    prediction_counts = {
        label: (prediction == label).sum() for label in np.unique(target)
    }
    class_precisions = {}
    for label in np.unique(target):
        indexes = prediction == label
        correct_predictions = (target[indexes] == prediction[indexes]).sum()
        class_precisions[label] = correct_predictions / prediction_counts[label]

    # print(class_precisions)

    return np.mean(list(class_precisions.values()))


def recall(
    prediction: np.ndarray[float], target: np.ndarray[float], strategy="weighted"
) -> float:
    class_counts = {label: (target == label).sum() for label in np.unique(target)}
    class_recalls = {}
    for label in np.unique(target):
        indexes = target == label
        correct_predictions = (target[indexes] == prediction[indexes]).sum()
        class_recalls[label] = correct_predictions / class_counts[label]

    # print(class_recalls)

    return np.mean(list(class_recalls.values()))


def accuracy(predicted_classes: np.ndarray, target_classes: np.ndarray) -> float:
    return (predicted_classes == target_classes).mean()


if __name__ == "__main__":
    df = pd.DataFrame.from_dict(
        {
            "A": [0, 1, 0, 1, 0, 1, 2, 1],
            "B": [1, 1, 1, 0, 1, 0, 1, 1],
            "Target": [1, 1, 1, 1, 0, 1, 0, 1],
        }
    )

    print(information_gain(df, "A", "Target"))
    print(gain_ratio(df, "Target", "Target"))
