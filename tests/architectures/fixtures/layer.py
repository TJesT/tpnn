from pytest import fixture
from random import randrange, choice, random
import numpy as np

from tpnn.networks import FCLayer, ACTLayer, LayerBase
from tpnn.networks import ActivationBuilder


@fixture
def const_array():
    return np.array([1, 0, 1, 1, 1])


@fixture
def const_actlayer():
    return FCLayer(
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
def fc_const_answer():
    return np.array([1.5, 2.5, 2.5])


@fixture(params=[(randrange(1, 50), randrange(1, 50)) for _ in range(10)])
def fc_layer(request):
    i, o = request.param
    return FCLayer(i, o)


@fixture(params=[None for _ in range(10)])
def act_layer(fc_layer: FCLayer):
    func = choice(ActivationBuilder().funcs)
    arg_names = ActivationBuilder().func_args(func)
    args = [random() for _ in range(len(arg_names))]
    activation = ActivationBuilder().build(func, args=args)
    return ACTLayer(
        fc_layer.output_dimension, fc_layer.output_dimension, activation=activation
    )


@fixture
def fc_array(fc_layer: LayerBase):
    return np.random.random(fc_layer.input_dimension) * 2 - 1


@fixture
def act_array(act_layer: LayerBase):
    return np.random.random(act_layer.input_dimension) * 2 - 1
