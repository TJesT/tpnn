from pytest import fixture

from tpnn.architectures.perceptron import Layer
from tpnn.architectures.perceptron.activation import ActivationBuilder
from random import randrange, choice, random
import numpy as np


@fixture
def const_array():
    return np.array([1, 0, 1, 1, 1])


@fixture
def const_layer():
    return Layer(
        input_dimension=5,
        output_dimension=3,
        weights=np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 1, 1],
            ]
        ),
        biases=[0.5, -0.5, 0.5],
    )


@fixture
def const_answer():
    return 1 / (1 + np.exp([-1.5, -2.5, -2.5]))


@fixture(params=[(randrange(1, 50), randrange(1, 50)) for _ in range(10)])
def layer(request):
    func = choice(ActivationBuilder().funcs)
    arg_names = ActivationBuilder().func_args(func)
    args = [random() for _ in range(len(arg_names))]
    activation = ActivationBuilder().build(func, args=args)
    return Layer(*request.param, activation=activation)


@fixture
def array(layer: Layer):
    return np.random.random(layer.input_dimension) * 2 - 1
