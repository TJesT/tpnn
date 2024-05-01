import numpy as np
from typing import Any, Generator, Iterable, overload, Literal, override, SupportsIndex
from dataclasses import dataclass

from tpnn.core.types import (
    BiasesGradients,
    WeightsGradients,
    BiasesGradient,
    WeightsGradient,
)

from .activation import Differentiable, AvailableActivations
from .layer import FCLayer, ACTLayer
from .loss import CEL, SEL
from .solver import GradientDescent
from .neural_network import NeuralNetwork


def batched(
    data: SupportsIndex, batch_size: int, length: int
) -> Generator[Any, None, None]:
    for start in range(0, length, batch_size):
        end = start + batch_size
        yield data[start:end]


@dataclass
class Perceptron(NeuralNetwork):
    layers: list[FCLayer | ACTLayer]

    def __post_init__(self):
        if isinstance(self.loss, str):
            if self.loss == "cross_entropy":
                self.loss = CEL
            elif self.loss == "squared_error":
                self.loss = SEL
            else:
                raise ValueError(f"No Loss with name {self.loss} were implemented")

        if isinstance(self.solver, str):
            if self.solver == "gradient_descent":
                self.solver = GradientDescent()
            else:
                raise ValueError(f"No Solver with name {self.solver} were implemented")

    def learn(
        self,
        data: np.ndarray[np.ndarray],
        targets: np.ndarray[np.ndarray],
        *,
        learn_rate: float = 0.1,
        momentum: float = 0.5,
        epochs: int = 20,
        batch_size: int = 1,
        while_not_learned: bool = False,
    ):
        if len(data.shape) < 2:
            raise ValueError(
                f"{len(data.shape) = }. "
                "If you want to learn per sample provide full dataset with `batch_size=1 (default)`."
            )
        if len(data.shape) < 3:
            raise ValueError(
                f"{len(data.shape) = }. Dataset should have 3 dimensions: (n_rows, dim1, dim2)."
                f"For singledimensional data reshape your dataset from (n_rows, dim) -> (n_rows, 1, dim)."
            )

        epochs += 1
        if isinstance(self.solver, GradientDescent):
            self.solver.learn_rate = learn_rate
            self.solver.momentum = momentum

        prev_cost = self.cost(data, targets)
        print(f"{prev_cost = }")
        while while_not_learned or (epochs := epochs - 1):
            for data_batch, target_batch in zip(
                batched(data, batch_size, data.shape[0]),
                batched(targets, batch_size, targets.shape[0]),
            ):
                self.update_weights(data_batch, target_batch)
                curr_cost = self.cost(data_batch, target_batch)
                if abs(curr_cost) < 10**-10:
                    self.to_file()
                    while_not_learned = False
                    epochs = 1
                    break
                # print("=" * 40)
                if abs(curr_cost) < abs(self.best_cost):
                    self.best_cost = curr_cost
                    self.save_backup()

                print(f"{curr_cost = }")

    def cost(self, data: np.ndarray, targets: np.ndarray) -> float:
        return np.mean(self.loss(targets, data >> self))

    def update_weights(self, data: np.ndarray[float], targets: np.ndarray[float]):
        data >> self
        gradients = self.get_gradients(targets)
        weight_updates, biases_updates = self.solver.get_updates(gradients)

        for fc, weight_update, biases_update in zip(
            self.layers[::2], weight_updates, biases_updates
        ):
            # print(f"{fc.weights=}")
            # print(f"{fc.biases=}")
            # print(f"{weight_update=}")
            # print(f"{biases_update=}")
            fc.weights += weight_update
            fc.biases += biases_update

    @override
    def get_gradients(
        self, target: np.ndarray
    ) -> tuple[WeightsGradients, BiasesGradients]:

        dC_dx = self._output_layer_biases_grads(target)
        biases_gradients = [dC_dx]
        weight_gradients = [self._output_layer_weights_grads(dC_dx)]

        for i in range(len(self.layers) - 3, -1, -2):
            dC_dx = self._other_layers_biases_grads(dC_dx, i)
            biases_gradients.append(dC_dx)

            dC_dw = self._other_layers_weights_grads(dC_dx, i)
            weight_gradients.append(dC_dw)

        weight_gradients = [
            np.mean(wgrads, axis=0) for wgrads in reversed(weight_gradients)
        ]
        biases_gradients = [
            np.mean(bgrads, axis=0) for bgrads in reversed(biases_gradients)
        ]

        return weight_gradients, biases_gradients

    def _output_layer_biases_grads(self, targets: np.ndarray[float]) -> BiasesGradient:
        dC_dActivation = self.loss.diff(targets, self.layers[-1].output)  # (k, 1, odim)
        # print(f"{dC_dActivation.shape = }")
        dActivation_dx = self.layers[-1].activation.diff(targets)  # (k, 1, odim)
        # print(f"{dActivation_dx.shape = }")
        dC_dx = dActivation_dx * dC_dActivation  # (k, 1, odim)
        # print(f"{dC_dx.shape = }")

        return dC_dx  # (k, 1, odim)

    def _output_layer_weights_grads(self, dC_dx: np.ndarray[float]) -> WeightsGradient:
        dC_dx = dC_dx  # (k, 1, odim)
        # print(f"{dC_dx.shape = }")
        activation_result = self.layers[-2].input  # (k, 1, idim)
        # print(f"{activation_result.shape = }")
        dx_dw = np.swapaxes(activation_result, 1, 2)  # (k, idim, 1)
        # print(f"{dx_dw.shape = }")
        dC_dw = dx_dw @ dC_dx  # (k, idim, odim)
        # print(f"{dC_dw.shape = }")

        return dC_dw  # (k, idim, odim)

    def _other_layers_biases_grads(
        self, dC_dx: np.ndarray[float], i: int
    ) -> BiasesGradient:
        dC_dx_old = dC_dx  # (k, 1, odim)
        # print(f"{dC_dx_old.shape = }")
        weights = self.layers[i + 1].weights  # (idim, odim)
        # print(f"{weights.shape = }")
        dx_old_da = weights.T  # (odim, idim)
        # print(f"{dx_old_da.shape = }")
        dC_da = dC_dx_old @ dx_old_da  # , (k, 1, idim)
        # print(f"{dC_da.shape = }")
        activation_diff = self.layers[i].activation.diff(
            self.layers[i].input
        )  # (k, 1, idim)
        # print(f"{activation_diff.shape = }")
        da_dx = activation_diff  # (k, 1, idim)
        # print(f"{da_dx.shape = }")
        dC_dx = dC_da * da_dx  # (k, 1, idim)
        # print(f"{dC_dx.shape = }")

        return dC_dx  # (k, 1, idim)

    def _other_layers_weights_grads(
        self, dC_dx: np.ndarray[float], i: int
    ) -> WeightsGradient:
        dC_dx  # (k, 1, odim)
        # print(f"{dC_dx.shape = }")
        activation_result = self.layers[i - 1].input  # (k, 1, idim)
        # print(f"{activation_result.shape = }")
        dx_dw = np.swapaxes(activation_result, 1, 2)  # (k, idim, 1)
        # print(f"{dx_dw.shape = }")
        dC_dw = dx_dw @ dC_dx  # (k, idim, odim)
        # print(f"{dC_dw.shape = }")

        return dC_dw

    @overload
    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        *,
        loss: Literal["cross_entropy", "squared_error"] = "squared_error",
        solver: Literal["gradient_descent"] = "gradient_descent",
        activations: list[Differentiable] = None,
    ) -> "Perceptron": ...
    @overload
    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        *,
        loss: Literal["cross_entropy", "squared_error"] = "squared_error",
        solver: Literal["gradient_descent"] = "gradient_descent",
        activations: list[AvailableActivations] = None,
        args_list: list[tuple[Any]] = None,
    ) -> "Perceptron": ...
    @overload
    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        *,
        loss: Literal["cross_entropy", "squared_error"] = "squared_error",
        solver: Literal["gradient_descent"] = "gradient_descent",
        activations: Differentiable = None,
    ) -> "Perceptron": ...
    @overload
    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        *,
        loss: Literal["cross_entropy", "squared_error"] = "squared_error",
        solver: Literal["gradient_descent"] = "gradient_descent",
        activations: AvailableActivations = None,
        args_list: tuple[Any] = None,
    ) -> "Perceptron": ...

    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        *,
        loss: Literal["cross_entropy", "squared_error"] = "squared_error",
        solver: Literal["gradient_descent"] = "gradient_descent",
        activations: Differentiable | list[Differentiable] = "relu",
        args_list: tuple[Any] | list[tuple[Any]] = None,
    ) -> "Perceptron":
        dims = list(dims)

        if any(not isinstance(size, int) for size in dims):
            raise TypeError("All sizes in dimensions should be python integers")
        if len(dims) < 2:
            raise ValueError("Dimensions should have 2 or more elements")

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

        return cls(
            layers=[
                (
                    lcls(idim, odim)
                    if lcls is FCLayer
                    else lcls(odim, odim, activation=activation, args=args)
                )
                for idim, odim, activation, args in zip(
                    dims, dims[1:], activations, args_list
                )
                for lcls in [FCLayer, ACTLayer]
            ],
            loss=loss,
            solver=solver,
        )


if __name__ == "__main__":
    data = np.loadtxt("./data/classification_test.txt")
    data = data[:, np.newaxis, :]

    print(f"{data.shape = }")

    def check_target(point: np.ndarray[float]) -> np.ndarray[float]:
        if np.linalg.norm(point) < 0.65:
            return [0, 0, 1]
        if np.linalg.norm(point - [0.5, 1]) < 0.45:
            return [0, 1, 0]
        return [1, 0, 0]

    target = np.array([check_target(point) for point in data])
    target = target[:, np.newaxis, :]
    print(f"{target.shape = }")

    from tpnn.core import set_lower_bound

    set_lower_bound(0.0)

    perceptron = Perceptron.from_dimensions(
        (2, 3),
        loss="cross_entropy",
        solver="gradient_descent",
        activations="leaky_relu",
        args_list=(0.05,),
    )
    perceptron.learn(
        data,
        target,
        learn_rate=0.1,
        momentum=0,
        batch_size=1000,
        # while_not_learned=True,
        epochs=3,
    )
