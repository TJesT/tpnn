import numpy as np
from enum import Enum
from itertools import cycle
from abc import ABC, abstractmethod
from typing import ClassVar, override
from dataclasses import dataclass, field

from tpnn.core.types import WeightsGradients, BiasesGradients


class _Type(Enum):
    BASE = "solver_base"
    GRADIENT_DESCENT = "gradient_descent"


@dataclass
class Solver(ABC):
    type: ClassVar[_Type] = field(default=_Type.BASE, init=False)

    @abstractmethod
    def get_updates(
        self,
        gradients: tuple[WeightsGradients, BiasesGradients],
    ) -> tuple[WeightsGradients, BiasesGradients]: ...


@dataclass
class GradientDescent(Solver):
    type: ClassVar[_Type] = field(default=_Type.GRADIENT_DESCENT, init=False)
    wvelocities: list[np.ndarray[float]] = field(default=cycle([0]), init=False)
    bvelocities: list[np.ndarray[float]] = field(default=cycle([0]), init=False)
    momentum: float = field(default=0.5, kw_only=True)
    learn_rate: float = field(default=0.1, kw_only=True)

    @override
    def get_updates(
        self,
        gradients: tuple[WeightsGradients, BiasesGradients],
    ) -> tuple[WeightsGradients, BiasesGradients]:
        weight_gradients, biases_gradients = gradients

        wupdates = [
            self.momentum * velocity - self.learn_rate * grad
            for velocity, grad in zip(self.wvelocities, weight_gradients)
        ]
        bupdates = [
            self.momentum * velocity - self.learn_rate * grad
            for velocity, grad in zip(self.bvelocities, biases_gradients)
        ]
        self.wvelocities = wupdates
        self.bvelocities = bupdates

        return wupdates, bupdates
