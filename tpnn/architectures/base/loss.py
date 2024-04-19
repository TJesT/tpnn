import numpy as np

from tpnn.core import Differentiable

Loss = Differentiable[np.ndarray, np.ndarray]


def squared_error_loss(target: np.ndarray, output: np.ndarray) -> np.ndarray:
    return np.power(output - target, 2)


def dsquared_error_loss(target: np.ndarray, output: np.ndarray) -> np.ndarray:
    return 2 * (output - target)


def cross_entropy_loss(target: np.ndarray, output: np.ndarray) -> np.ndarray:
    return -target * np.log2(output + 0.001)


def dcross_entropy_loss(target: np.ndarray, output: np.ndarray) -> np.ndarray:
    return -target / output


SEL = Loss(squared_error_loss, dsquared_error_loss)
CEL = Loss(cross_entropy_loss, dcross_entropy_loss)
