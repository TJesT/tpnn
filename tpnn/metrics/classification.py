import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.types import Label, Probabilty

type FalsePositiveRates = np.ndarray[float]
type TruePositiveRates = np.ndarray[float]


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


def fpr_tpr(
    predicted: np.ndarray[float], target: np.ndarray[float]
) -> tuple[FalsePositiveRates, TruePositiveRates]:
    threshold_values = sorted(np.unique(predicted), reverse=True)
    tpr_values = [0]
    fpr_values = [0]
    num_positive = np.sum(target)
    num_negative = len(target) - num_positive

    for threshold in threshold_values:
        predicted_binary = np.where(predicted >= threshold, 1, 0)
        true_positives = np.sum((predicted_binary == 1) & (target == 1))
        false_positives = np.sum((predicted_binary == 1) & (target == 0))
        tpr = true_positives / num_positive
        fpr = false_positives / num_negative
        tpr_values.append(tpr)
        fpr_values.append(fpr)

    return fpr_values, tpr_values


def roc_auc(predicted: np.ndarray[float], target: np.ndarray[float]):
    classes_count = target.shape[2]
    clasess_fpr = np.linspace(0, 1, 100)
    classes_tpr = []

    auc = lambda fpr, tpr: np.trapz(tpr, fpr)

    plt.figure()
    for class_index in range(classes_count):
        fpr, tpr = fpr_tpr(predicted[:, :, class_index], target[:, :, class_index])
        plt.plot(fpr, tpr, label=f"Class {class_index} (AUC = {auc(fpr, tpr):.3f})")
        interpolated_tpr = np.interp(clasess_fpr, fpr, tpr)
        classes_tpr.append(interpolated_tpr)

    micro_mean_tpr = np.mean(classes_tpr, axis=0)
    macro_mean_tpr = np.sum(
        [
            tpr * np.sum(target[:, :, class_i], axis=0)[0] / target.shape[0]
            for class_i, tpr in enumerate(classes_tpr)
        ],
        axis=0,
    )

    plt.plot(
        clasess_fpr,
        micro_mean_tpr,
        label=f"Micro-average ROC (AUC = {auc(clasess_fpr, micro_mean_tpr):.3f})",
        linestyle="--",
    )
    plt.plot(
        clasess_fpr,
        macro_mean_tpr,
        label=f"Macro-average ROC (AUC = {auc(clasess_fpr, macro_mean_tpr):.3f})",
        linestyle="--",
    )
    padding = 0.02
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0 - padding, 1.0])
    plt.ylim([0.0, 1.0 + padding])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")

    return plt


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
