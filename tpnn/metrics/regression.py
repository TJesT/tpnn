import numpy as np


def r2_score(prediction: np.ndarray[float], target: np.ndarray[float]) -> float:
    mean_true = np.mean(target)
    total_ss = np.sum(np.square(target - mean_true))
    resid_ss = np.sum(np.square(target - prediction))
    return 1 - (resid_ss / total_ss)


def mean_absolute_error(
    prediction: np.ndarray[float], target: np.ndarray[float]
) -> float:
    return np.sum(np.abs(target - prediction)) / target.shape[0]


def root_mean_squared_error(
    prediction: np.ndarray[float], target: np.ndarray[float]
) -> float:
    return np.sqrt(np.sum(np.square(target - prediction)) / target.shape[0])
