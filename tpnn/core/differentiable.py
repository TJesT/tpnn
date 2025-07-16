import numpy as np
from dataclasses import dataclass, field
from typing import Callable, ClassVar
from .types import Probabilty


@dataclass(frozen=True)
class Differentiable[*Inputs]:
    UpperBound: ClassVar[np.number] = field(default=+512, init=False)
    LowerBound: ClassVar[np.number] = field(default=-512, init=False)
    func: Callable[[*Inputs], np.ndarray[Probabilty]]
    diff: Callable[[*Inputs], np.ndarray[float]]

    def __call__(self, *inputs: *Inputs) -> np.ndarray[Probabilty]:
        return np.clip(self.func(*inputs), self.LowerBound, self.UpperBound)


def set_lower_bound(lower_bound: np.number):
    Differentiable.LowerBound = lower_bound


def set_upper_bound(upper_bound: np.number):
    Differentiable.UpperBound = upper_bound
