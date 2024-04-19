import numpy as np
from dataclasses import dataclass
from typing import Any, Iterable, overload, override, Literal

from .base.activation import Differentiable, ActivationBuilder, sigmoid
from .base.layer import FCLayer, ACTLayer
from .base.loss import Loss, CEL, SEL
from tpnn.core import Pipeable


@dataclass
class Perceptron(Pipeable[np.ndarray[np.number], np.ndarray[np.number]]):
    layers: list[FCLayer | ACTLayer]
    loss: Loss

    @override
    def __call__(self, _input: np.ndarray[np.number]) -> np.ndarray[np.number]:
        if (ishape := _input.shape[0]) != (lshape := self.layers[0].input_dimension):
            raise ValueError(
                f"Shape of input={ishape} doesn't fit first layer's shape={lshape}"
            )

        curr = _input
        for i, layer in enumerate(self.layers, 1):
            curr = curr >> layer

        return curr

    def get_gradients(
        self, target: np.ndarray[np.number]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # taget: 2
        # dC_da: 2
        dC_dActivation = self.loss.diff(
            target, self.layers[-1].output
        )  # nodeLoss derivative
        # da_dx: 2
        dActivation_dx = self.layers[-1].activation.diff(target)
        # dC_dx: 2
        dC_dx = dActivation_dx * dC_dActivation  # nodeValues

        # activation: (3, 1)
        activation_result = self.layers[-2].input[:, np.newaxis]
        # dx_dw: (3, 2)
        dx_dw = activation_result @ dC_dx[np.newaxis, :]
        # dC_dw: (3, 2)
        dC_dw = dx_dw * dC_dx

        weight_gradients = [dC_dw.copy()]
        biases_gradients = [dC_dx.copy()]  # also list of all nodeValues

        # layers: input(2) (2, 3), (3, 3), (3, 2), (2, 2) output(2) target(2)

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

    def epoch(
        self,
        data: list[np.ndarray],
        targets: list[np.ndarray],
        *,
        learn_rate: float = 0.1,
    ):
        weights_gradients = [
            np.zeros((layer.input_dimension, layer.output_dimension))
            for layer in self.layers[::2]
        ]
        biases_gradients = [
            np.zeros(layer.output_dimension) for layer in self.layers[::2]
        ]

        for row, target in zip(data, targets):
            row >> self
            row_weight_gradients, row_biases_gradients = self.get_gradients(target)

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

        # print(weights_gradients)
        # print(biases_gradients)

        for fc, weights_gradient, biases_gradient in zip(
            self.layers[::2], weights_gradients, biases_gradients
        ):
            fc.weights -= learn_rate * weights_gradient
            fc.biases -= learn_rate * biases_gradient

    def learn(
        self,
        data: list[np.ndarray],
        targets: list[np.ndarray],
        *,
        learn_rate: float = 0.1,
        epochs: int = 1,
    ):
        print(f"{self.cost(data, targets) = }")
        for _ in range(epochs):
            self.epoch(data, targets, learn_rate=learn_rate)
            print(f"{self.cost(data, targets) = }")

    def cost(self, data: list[np.ndarray], targets: list[np.ndarray]) -> float:
        return np.mean(
            [self.loss(target, row >> self) for row, target in zip(data, targets)]
        )

    @overload
    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        loss: Literal["cross_entropy", "squared_error"],
        *,
        activations: list[Differentiable] = None,
    ) -> "Perceptron": ...
    @overload
    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        loss: Literal["cross_entropy", "squared_error"],
        *,
        activations: list[Literal["sigmoid", "heaviside", "softmax", "relu"]] = None,
        args_list: list[tuple[Any]] = None,
    ) -> "Perceptron": ...
    @overload
    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        loss: Literal["cross_entropy", "squared_error"],
        *,
        activations: Differentiable = None,
    ) -> "Perceptron": ...
    @overload
    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        loss: Literal["cross_entropy", "squared_error"],
        *,
        activations: Literal["sigmoid", "heaviside", "softmax", "relu"] = None,
        args_list: list[tuple[Any]] = None,
    ) -> "Perceptron": ...

    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        loss: Literal["cross_entropy", "squared_error"],
        *,
        activations: Differentiable | list[Differentiable] = None,
        args_list: tuple[Any] | list[tuple[Any]] = None,
    ) -> "Perceptron":
        dims = list(dims)

        if any(not isinstance(size, int) for size in dims):
            raise TypeError("All sizes in dimensions should be python integers")
        if len(dims) < 2:
            raise ValueError("Dimensions should have 2 or more elements")

        if loss == "cross_entropy":
            loss = CEL
        else:
            loss = SEL

        if activations is None:
            activations = sigmoid
        if args_list is None:
            args_list = ()

        if not isinstance(activations, list):
            activations = [activations] * (len(dims) - 1)
        if not isinstance(args_list, list):
            args_list = [args_list] * (len(dims) - 1)

        if len(activations) != len(dims) - 1:
            raise ValueError(
                "List of Activations should have `len(Dimensions) - 1` elements. "
                f"But you have {len(activations) = } instead of len(activation) = {len(dims)-1}"
            )

        activations = [
            ActivationBuilder().build(act, args=args) if isinstance(act, str) else act
            for act, args in zip(activations, args_list)
        ]

        print(activations)

        return cls(
            layers=[
                (
                    lcls(idim, odim)
                    if lcls is FCLayer
                    else lcls(odim, odim, activation=activation)
                )
                for idim, odim, activation in zip(dims, dims[1:], activations)
                for lcls in [FCLayer, ACTLayer]
            ],
            loss=loss,
        )


if __name__ == "__main__":
    perceptron = Perceptron.from_dimensions(
        (3, 4, 3), "cross_entropy", activations=["sigmoid", "softmax"]
    )
    _input = np.array([0, 0.5, 0.3])
    data = [np.random.random(size=3) for _ in range(100)]
    targets = [
        (
            np.array([0, 0, 1])
            if (rsum := np.sum(row)) > 1
            else (np.array([0, 1, 0]) if rsum > 0.5 else np.array([1, 0, 0]))
        )
        for row in data
    ]

    print(_input, "->", _input >> perceptron)

    perceptron.learn(data, targets, learn_rate=0.6, epochs=10)

    _input = np.array([0, 0.5, 0.3])
    print(_input, "->", _input >> perceptron)
    _input = np.array([0.5, 0.4, 0.2])
    print(_input, "->", _input >> perceptron)
    _input = np.array([0.1, 0, 0.2])
    print(_input, "->", _input >> perceptron)
    _input = np.array([0.2, 0.3, 0.1])
    print(_input, "->", _input >> perceptron)
