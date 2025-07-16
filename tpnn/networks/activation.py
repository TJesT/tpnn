import numpy as np
from typing import Any, Literal
from tpnn.core.types import Probabilty
from tpnn.core.singleton import Singleton
from tpnn.core.differentiable import Differentiable

Activation = Differentiable[np.ndarray[float]]
type AvailableActivations = Literal[
    "sigmoid", "softmax", "relu", "tanh", "leaky_relu", "heaviside"
]


# ===========================#==================#=========================== #
# # ======================= # Activation Getters # ======================= # #
# ===========================#==================#=========================== #


def get_heaviside(thresh: float) -> Activation:
    def dheaviside(_input: np.ndarray[float]) -> np.ndarray[float]:
        if len(_input.shape) > 2:
            return np.apply_along_axis(dheaviside, axis=2, arr=_input)
        return np.where(_input == thresh, np.inf, 0)

    def heaviside(_input: np.ndarray[float]) -> np.ndarray[Probabilty]:
        if len(_input.shape) > 2:
            return np.apply_along_axis(heaviside, axis=2, arr=_input)
        return np.greater(_input, thresh).astype(float)

    return Activation(heaviside, dheaviside)


def get_leaky_relu(alpha: float = 0.05) -> Activation:
    def relu_activation(_input: np.ndarray[float]) -> np.ndarray[float]:
        if len(_input.shape) > 2:
            return np.apply_along_axis(relu_activation, axis=2, arr=_input)
        return np.where(_input > 0, _input, alpha * _input)

    def drelu_activation(_input: np.ndarray[float]) -> np.ndarray[float]:
        if len(_input.shape) > 2:
            return np.apply_along_axis(drelu_activation, axis=2, arr=_input)
        return np.where(_input > 0, 1, alpha)

    return Activation(relu_activation, drelu_activation)


# ===========================#==================#=========================== #
# # ======================== # Pure Activations # ======================== # #
# ===========================#==================#=========================== #


relu = get_leaky_relu(alpha=0)


def softmax_activation(_input: np.ndarray[float]) -> np.ndarray[Probabilty]:
    if len(_input.shape) > 2:
        return np.apply_along_axis(softmax_activation, axis=2, arr=_input)
    e_x = np.exp(_input - np.max(_input))
    return e_x / np.sum(e_x, axis=0)


def dsoftmax_activation(_input: np.ndarray[float]) -> np.ndarray[float]:
    if len(_input.shape) > 2:
        return np.apply_along_axis(dsoftmax_activation, axis=2, arr=_input)
    softmaxed_input = softmax_activation(_input).reshape((-1, 1))
    return _input @ (
        np.diagflat(softmaxed_input) - np.dot(softmaxed_input, softmaxed_input.T)
    )


softmax = Activation(softmax_activation, dsoftmax_activation)


def tanh_activation(_input: np.ndarray[float]) -> np.ndarray[Probabilty]:
    if len(_input.shape) > 2:
        return np.apply_along_axis(tanh_activation, axis=2, arr=_input)

    return np.tanh(_input)


def dtanh_activation(_input: np.ndarray[float]) -> np.ndarray[float]:
    if len(_input.shape) > 2:
        return np.apply_along_axis(dtanh_activation, axis=2, arr=_input)

    return 1 - np.square(np.tanh(_input))


tanh = Activation(tanh_activation, dtanh_activation)


def sigmoid_activation(_input: np.ndarray[float]) -> np.ndarray[Probabilty]:
    if len(_input.shape) > 2:
        return np.apply_along_axis(sigmoid_activation, axis=2, arr=_input)

    positives = _input >= 0
    negatives = ~positives

    exp_x_neg = np.exp(_input[negatives])

    sigmoid_result = _input.copy()
    sigmoid_result[positives] = 1 / (1 + np.exp(-_input[positives]))
    sigmoid_result[negatives] = exp_x_neg / (1 + exp_x_neg)

    return sigmoid_result


def dsigmoid_activation(_input: np.ndarray[float]) -> np.ndarray[float]:
    if len(_input.shape) > 2:
        return np.apply_along_axis(dsigmoid_activation, axis=2, arr=_input)
    sigmoid_value = sigmoid_activation(_input)
    return sigmoid_value * (1 - sigmoid_value)


sigmoid = Activation(sigmoid_activation, dsigmoid_activation)


# ===========================#==================#=========================== #
# # ======================= # Activation Builder # ======================= # #
# ===========================#==================#=========================== #


class ActivationBuilder(metaclass=Singleton):
    def __init__(self) -> None:
        self.__map = {
            "sigmoid": sigmoid,
            "softmax": softmax,
            "relu": relu,
            "tanh": tanh,
            "leaky_relu": get_leaky_relu,
            "heaviside": get_heaviside,
        }

    def build(
        self,
        func: AvailableActivations,
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
        self,
        func: AvailableActivations,
    ) -> tuple[str]:
        func = self.__map.get(func)
        if isinstance(func, Differentiable):
            func = func.func
        args = func.__code__.co_varnames[: func.__code__.co_argcount]
        if "_input" in args:
            return ()
        return args

    @property
    def funcs(self):
        return tuple(self.__map.keys())


if __name__ == "__main__":
    data = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 0, 1],
        ]
    )
    print(softmax_activation(data[0]))
    print(softmax_activation(data))
