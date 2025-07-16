from dataclasses import dataclass, field
import numpy as np
from typing import override

from tpnn.networks.layer import (
    LayerBase,
    FCLayer,
    ACTLayer,
    CONVLayer,
    POOLLayer,
    FLATLayer,
)
from tpnn.networks.neural_network import (
    NeuralNetwork,
    WeightsGradients,
    BiasesGradients,
)


def __lenet_layers_init__() -> list[LayerBase]:
    return [
        CONVLayer((1, 28, 28), (6, 28, 28), kernel_size=5, depth=6, padding=2),
        ACTLayer((6, 28, 28), activation="sigmoid"),
        POOLLayer((6, 28, 28), pool_type="mean", pool_size=2, stride=2),
        CONVLayer((6, 14, 14), (16, 10, 10), kernel_size=5, depth=16),
        ACTLayer((16, 10, 10), activation="sigmoid"),
        POOLLayer((16, 10, 10), pool_type="mean", pool_size=2, stride=2),
        FLATLayer((16, 5, 5), 400),
        FCLayer(400, 120),
        ACTLayer(120, activation="sigmoid"),
        FCLayer(120, 84),
        ACTLayer(84, activation="sigmoid"),
        FCLayer(84, 10),
        FLATLayer((1, 10), 10),
        ACTLayer(10, activation="softmax"),
    ]


def __lenet_layers_simplified__() -> list[LayerBase]:
    return [
        CONVLayer((1, 28, 28), (5, 24, 24), 5, 5),
        ACTLayer((5, 24, 24), activation="sigmoid"),
        POOLLayer((5, 24, 24), 2),
        CONVLayer((5, 12, 12), (3, 8, 8), 5, 3),
        ACTLayer((3, 8, 8), activation="sigmoid"),
        FLATLayer((3, 8, 8), 3 * 8 * 8),
        FCLayer(3 * 8 * 8, 32),
        ACTLayer(32, activation="sigmoid"),
        FCLayer(32, 10),
        FLATLayer((1, 10), 10),
        ACTLayer(10, activation="softmax"),
    ]


@dataclass
class LeNet(NeuralNetwork):
    layers: list[LayerBase] = field(
        init=False, default_factory=__lenet_layers_simplified__
    )

    @override
    def get_gradients(
        self, target: np.ndarray
    ) -> tuple[WeightsGradients, BiasesGradients]: ...

    @override
    def update_weights(self, data: np.ndarray, target: np.ndarray) -> None: ...
