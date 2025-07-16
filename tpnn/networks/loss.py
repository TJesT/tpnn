from dataclasses import dataclass, field
from typing import ClassVar
import numpy as np


from tpnn.core import Differentiable


@dataclass(frozen=True)
class Loss(Differentiable[np.ndarray, np.ndarray]):
    LowerBound: ClassVar[float] = field(default=-Differentiable.UpperBound, init=False)


def squared_error_loss(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return 0.5 * np.mean(np.power(predicted - target, 2), axis=2)


def dsquared_error_loss(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return predicted - target


def cross_entropy_loss(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return -np.sum(target * np.log2(predicted + 0.001), axis=2)


def dcross_entropy_loss(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return predicted - target


SEL = Loss(squared_error_loss, dsquared_error_loss)
CEL = Loss(cross_entropy_loss, dcross_entropy_loss)
