import numpy as np
from dataclasses import dataclass
from typing import Callable, TypeVar, TypeVarTuple
from .types import Probabilty

Input = TypeVar("Input", bound=np.ndarray[float])
Inputs = TypeVarTuple("Inputs")


@dataclass(frozen=True)
class Differentiable:
    func: Callable[[Input, *Inputs], np.ndarray[Probabilty]]
    diff: Callable[[Input, *Inputs], np.ndarray[float]]

    def __call__(self, _input: np.ndarray[float], *inputs) -> np.ndarray[Probabilty]:
        return self.func(_input, *inputs)
