from typing import Any, Callable, Literal
import numpy as np

from tpnn.core.types import Probabilty
from tpnn.core.singleton import Singleton

type Activation = Callable[[float], Probabilty]

# TODO:
# def softmax(_input: float) -> Probabilty:
#     return
# TODO:
# def argmax(_input: float) -> Probabilty:
#     return


def sigmoid(_input: float) -> Probabilty:
    return 1 / (1 + np.exp(-_input))


def get_heaviside_func(thresh: float):
    def heaviside(_input: float) -> Probabilty:
        return float(_input > thresh)

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
        func = self.__map.get(func)
        if varnames := set(func.__code__.co_varnames) - {"_input"}:
            if len(args) != len(varnames):
                raise TypeError(
                    "Provide exact number of needed arguments. "
                    f"ActivationBuilder.build(..., args={varnames})"
                )
            func = func(*args)
        return func
