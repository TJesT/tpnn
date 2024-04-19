import numpy as np
from dataclasses import dataclass
from typing import Callable, TypeVar, TypeVarTuple
from .types import Probabilty


@dataclass(frozen=True)
class Differentiable[*Inputs]:
    func: Callable[[*Inputs], np.ndarray[Probabilty]]
    diff: Callable[[*Inputs], np.ndarray[float]]

    def __call__(self, *inputs: *Inputs) -> np.ndarray[Probabilty]:
        return self.func(*inputs)
