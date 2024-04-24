import numpy as np
from typing import Any, Literal
from tpnn.core.types import Probabilty
from tpnn.core.singleton import Singleton
from tpnn.core.differentiable import Differentiable

Activation = Differentiable[np.ndarray[float]]


def softmax_activation(_input: np.ndarray[float]) -> np.ndarray[Probabilty]:
    e_x = np.exp(_input - np.max(_input))
    return e_x / np.sum(e_x, axis=0)


def dsoftmax_activation(_input: np.ndarray[float]) -> np.ndarray[float]:
    softmaxed_input = softmax_activation(_input).reshape((-1, 1))
    return _input @ (
        np.diagflat(softmaxed_input) - np.dot(softmaxed_input, softmaxed_input.T)
    )


softmax = Activation(softmax_activation, dsoftmax_activation)


def relu_activation(_input: np.ndarray[float]) -> np.ndarray[float]:
    return np.where(_input > 0, _input, 0)


def drelu_activation(_input: np.ndarray[float]) -> np.ndarray[float]:
    return np.where(_input > 0, 1, 0)


relu = Activation(relu_activation, drelu_activation)


def sigmoid_activation(_input: np.ndarray[float]) -> np.ndarray[Probabilty]:
    positives = _input >= 0
    negatives = ~positives

    exp_x_neg = np.exp(_input[negatives])

    sigmoid_result = _input.copy()
    sigmoid_result[positives] = 1 / (1 + np.exp(-_input[positives]))
    sigmoid_result[negatives] = exp_x_neg / (1 + exp_x_neg)

    return sigmoid_result


def dsigmoid_activation(_input: np.ndarray[float]) -> np.ndarray[float]:
    sigmoid_value = sigmoid_activation(_input)
    return sigmoid_value * (1 - sigmoid_value)


sigmoid = Activation(sigmoid_activation, dsigmoid_activation)


def get_heaviside(thresh: float):
    def dheaviside(_input: np.ndarray[float]) -> np.ndarray[float]:
        return np.where(_input == thresh, np.inf, 0)

    def heaviside(_input: np.ndarray[float]) -> np.ndarray[Probabilty]:
        return np.greater(_input, thresh).astype(float)

    return Activation(heaviside, dheaviside)


class ActivationBuilder(metaclass=Singleton):
    def __init__(self) -> None:
        self.__map = {
            "sigmoid": sigmoid,
            "softmax": softmax,
            "relu": relu,
            "heaviside": get_heaviside,
        }

    def build(
        self,
        func: Literal["sigmoid", "heaviside", "softmax", "relu"],
        *,
        args: tuple[Any] = (),
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

    def func_args(
        self, func: Literal["sigmoid", "heaviside", "softmax", "relu"]
    ) -> tuple[str]:
        func = self.__map.get(func)
        if isinstance(func, Differentiable):
            func = func.func
        args = func.__code__.co_cellvars
        if "_input" in args:
            return ()
        return args

    @property
    def funcs(self):
        return tuple(self.__map.keys())


print(f"{isinstance(relu, Differentiable) = }")
