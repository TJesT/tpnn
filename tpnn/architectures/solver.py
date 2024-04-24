from enum import Enum
from typing import ClassVar, override
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from tpnn.architectures.layer import ACTLayer, FCLayer
from tpnn.architectures.loss import Loss
from tpnn.core.pipeline import Pipeable

type BiasesGradients = list[np.ndarray[float]]
type WeightsGradients = list[np.ndarray[float]]
type NeuralNetwork = Pipeable[np.ndarray[float], np.ndarray[float]]


class _Type(Enum):
    BASE = "solver_base"
    GRADIENT_DESCENT = "gradient_descent"


@dataclass
class Solver(ABC):
    type: ClassVar[_Type] = field(default=_Type.BASE, init=False)
    layers: list[FCLayer | ACTLayer]
    loss: Loss
    learn_rate: float = 0.1

    @abstractmethod
    def calculate_gradients(
        self, target: np.ndarray[float]
    ) -> tuple[WeightsGradients, BiasesGradients]: ...

    def update_gradients(
        self, network: NeuralNetwork, data: np.ndarray[float], targets
    ):
        weights_gradients = [
            np.zeros((layer.input_dimension, layer.output_dimension))
            for layer in self.layers[::2]
        ]
        biases_gradients = [
            np.zeros(layer.output_dimension) for layer in self.layers[::2]
        ]

        for row, target in zip(data, targets):
            row >> network
            row_weight_gradients, row_biases_gradients = self.calculate_gradients(
                target
            )

            for (
                weights_gradient,
                row_weight_gradient,
                biases_gradient,
                row_biases_gradient,
            ) in zip(
                weights_gradients,
                row_weight_gradients,
                biases_gradients,
                row_biases_gradients,
            ):
                weights_gradient += row_weight_gradient
                biases_gradient += row_biases_gradient

        for fc, weights_gradient, biases_gradient in zip(
            self.layers[::2], weights_gradients, biases_gradients
        ):
            fc.weights -= self.learn_rate * weights_gradient / len(data)
            fc.biases -= self.learn_rate * biases_gradient / len(data)


@dataclass
class GradientDescent(Solver):
    type: ClassVar[_Type] = field(default=_Type.GRADIENT_DESCENT, init=False)
    velocities: list[np.ndarray[float]] = field(default=None, init=False)
    momentum: float = field(default=0.9, kw_only=True)

    def __post_init__(self):
        self.velocities = [np.zeros_like(layer.weights) for layer in self.layers[::2]]

    @override
    def calculate_gradients(
        self, target: np.ndarray[float]
    ) -> tuple[WeightsGradients, BiasesGradients]:
        dC_dActivation = self.loss.diff(target, self.layers[-1].output)
        dActivation_dx = self.layers[-1].activation.diff(target)
        dC_dx = dActivation_dx * dC_dActivation

        activation_result = self.layers[-2].input[:, np.newaxis]
        dx_dw = activation_result @ dC_dx[np.newaxis, :]
        dC_dw = dx_dw * dC_dx

        weight_gradients = [dC_dw.copy()]
        biases_gradients = [dC_dx.copy()]

        for i in range(len(self.layers) - 3, -1, -2):
            dC_dx_old = dC_dx  # (2, 1)
            dx_old_da = self.layers[i + 1].weights  # (3, 2)
            dC_da = dx_old_da @ dC_dx_old  # 3
            da_dx = self.layers[i].activation.diff(self.layers[i].input)  # 3
            dC_dx = da_dx * dC_da  # 3
            activation_result = self.layers[i - 1].input[:, np.newaxis]  # (2, 1)
            dC_dw = activation_result @ dC_dx[np.newaxis, :]  # (2, 3)
            weight_gradients.append(dC_dw.copy())
            biases_gradients.append(dC_dx.copy())

        return weight_gradients[::-1], biases_gradients[::-1]
