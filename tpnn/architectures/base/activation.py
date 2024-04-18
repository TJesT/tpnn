import numpy as np
from typing import Any, Literal
from tpnn.core.types import Probabilty
from tpnn.core.singleton import Singleton
from tpnn.core.differentiable import Differentiable


def softmax_activation(_input: np.ndarray[float]) -> np.ndarray[Probabilty]:
    e_x = np.exp(_input - np.max(_input))
    return e_x / np.sum(e_x, axis=0)


softmax = Differentiable(softmax_activation, None)

# TODO:
# def relu(_input: float) -> Probabilty:
#     return


def dsigmoid_activation(_input: np.ndarray[float]) -> np.ndarray[float]:
    sigmoid_value = sigmoid_activation(_input)
    return sigmoid_value * (1 - sigmoid_value)


def sigmoid_activation(_input: np.ndarray[float]) -> np.ndarray[Probabilty]:
    return 1 / (1 + np.exp(-_input))


sigmoid = Differentiable(sigmoid_activation, dsigmoid_activation)


def get_heaviside_func(thresh: float):
    def dheaviside(_input: np.ndarray[float]) -> np.ndarray[float]:
        return np.where(_input == thresh, np.inf, 0)

    def heaviside(_input: np.ndarray[float]) -> np.ndarray[Probabilty]:
        return np.greater(_input, thresh).astype(float)

    return Differentiable(heaviside, dheaviside)


class ActivationBuilder(metaclass=Singleton):
    def __init__(self) -> None:
        self.__map = {
            "sigmoid": sigmoid,
            "heaviside": get_heaviside_func,
            "softmax": softmax,
        }

    def build(
        self, func: Literal["sigmoid", "heaviside", "softmax"], *, args: tuple[Any] = ()
    ) -> Differentiable:
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

    def func_args(self, func: Literal["sigmoid", "heaviside", "softmax"]) -> tuple[str]:
        func = self.__map.get(func)
        args = func.__code__.co_cellvars
        if "_input" in args:
            return ()
        return args

    @property
    def funcs(self):
        return tuple(self.__map.keys())
