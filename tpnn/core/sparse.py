from functools import wraps
from typing import Callable, Iterable, Optional, Sequence, SupportsIndex, overload
import numpy as np
from numpy._typing import NDArray


type Value = np.number
type Values = Iterable[Value]
type Index = int
type Indexes = Iterable[Index]


class SparseVector(np.ndarray):
    def __new__(cls: "SparseVector", array) -> "SparseVector":
        obj = np.asarray(tuple((i, v) for i, v in enumerate(array) if v)).view(cls)

        return obj

    def __mul_sparse(self, other: "SparseVector") -> np.number: ...
