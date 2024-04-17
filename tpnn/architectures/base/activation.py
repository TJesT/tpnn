from typing import Any, Callable, Literal
import numpy as np

from tpnn.core.types import Probabilty
from tpnn.core.singleton import Singleton

type Activation = (
    Callable[[np.ndarray[float]], np.ndarray[Probabilty]]
    | Callable[[float], Probabilty]
)

# TODO:
# def softmax(_input: float) -> Probabilty:
#     return
# TODO:
# def argmax(_input: float) -> Probabilty:
#     return

# TODO:
# def relu(_input: float) -> Probabilty:
#     return


def sigmoid(_input: np.ndarray[float]) -> np.ndarray[Probabilty]:
    return 1 / (1 + np.exp(-_input))


def get_heaviside_func(thresh: float):
    def heaviside(_input: np.ndarray[float]) -> np.ndarray[Probabilty]:
        return np.greater(_input, thresh).astype(float)

    return heaviside


class ActivationBuilder(metaclass=Singleton):
    def __init__(self) -> None:
        self.__map = {
            "sigmoid": sigmoid,
            "heaviside": get_heaviside_func,
        }

    def build(
        self, func: Literal["sigmoid", "heaviside"], *, args: tuple[Any] = ()
    ) -> Activation:
        varnames = self.func_args(func)
        func = self.__map.get(func)

        if varnames:
            if len(args) != len(varnames):
                raise TypeError(
                    "Provide exact number of needed arguments. "
                    f"ActivationBuilder.build(..., args={varnames}). "
                    f"But you provided {args=}"
                )
            func = func(*args)

        return func

    def func_args(self, func: Literal["sigmoid", "heaviside"]) -> tuple[str]:
        func = self.__map.get(func)
        args = func.__code__.co_cellvars
        if "_input" in args:
            return ()
        return args

    @property
    def funcs(self):
        return tuple(self.__map.keys())
