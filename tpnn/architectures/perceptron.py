import json
import numpy as np
from copy import copy
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, overload, override, Literal

from .activation import Differentiable, ActivationBuilder, sigmoid
from .layer import FCLayer, ACTLayer, LayerDeserializer, LayerSerializer
from .loss import Loss, CEL, SEL
from .solver import Solver, GradientDescent
from tpnn.core import Pipeable


class NeuralNetworkSerializer(LayerSerializer):
    def default(self, o: Any) -> Any:
        if isinstance(o, Perceptron):
            return {
                "layers": o.layers,
                "loss": self.default(o.loss),
                "solver": self.default(o.solver),
            }
        if isinstance(o, Solver):
            return {"type": o.type.value, "learn_rate": o.learn_rate}
        if isinstance(o, Loss):
            if o.func.__name__ == "squared_error_loss":
                return {"type": "loss", "name": "squared_error"}
            if o.func.__name__ == "cross_entropy_loss":
                return {"type": "loss", "name": "cross_entropy"}
            return None

        return super().default(o)


class NeuralNetworkDeserializer(LayerDeserializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, object_hook=self.object_hook, **kwargs)

    def object_hook(self, _dict):
        match _dict:
            case {
                "layers": layers,
                "loss": loss,
                "solver": solver,
            }:
                solver.layers = layers
                solver.loss = loss
                return Perceptron(layers=layers, loss=loss, solver=solver)
            case {"type": "loss", "name": loss_name}:
                if loss_name == "squared_error":
                    return SEL
                if loss_name == "cross_entropy":
                    return CEL
            case {"type": solver_type, "learn_rate": learn_rate}:
                if solver_type == "gradient_descent":
                    return GradientDescent(None, None, learn_rate)
            case _:
                return LayerDeserializer.object_hook(self, _dict)


@dataclass
class Perceptron(Pipeable[np.ndarray[np.number], np.ndarray[np.number]]):
    layers: list[FCLayer | ACTLayer]
    loss: Loss
    solver: Solver
    best_cost: float = field(init=False, default=float("+inf"))
    backup_layers: list[FCLayer | ACTLayer] = field(init=False, default=None)

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
                self.solver = GradientDescent(self.layers, loss=self.loss)
            else:
                raise ValueError(f"No Solver with name {self.solver} were implemented")

    @override
    def __call__(self, _input: np.ndarray[np.number]) -> np.ndarray[np.number]:
        if (ishape := _input.shape[0]) != (lshape := self.layers[0].input_dimension):
            raise ValueError(
                f"Shape of input={ishape} doesn't fit first layer's shape={lshape}"
            )

        curr = _input
        for _, layer in enumerate(self.layers, 1):
            curr = curr >> layer

        return curr

    def learn(
        self,
        data: list[np.ndarray],
        targets: list[np.ndarray],
        *,
        learn_rate: float = 0.1,
        epochs: int = 20,
        while_not_learned: bool = False,
    ):
        epochs += 1
        self.solver.learn_rate = learn_rate
        prev_cost = self.cost(data, targets)
        print(f"{prev_cost = }")
        while while_not_learned or (epochs := epochs - 1):
            self.solver.update_gradients(self, data, targets)
            curr_cost = self.cost(data, targets)
            if abs(curr_cost) > self.best_cost * 1.2:
                self.load_backup()
                break
            elif abs(curr_cost) < 10**-10:
                self.to_file()
                break
            elif abs(curr_cost) < self.best_cost:
                self.save_backup()
                self.best_cost = abs(curr_cost)
                print(f"{curr_cost = }")

    def cost(self, data: list[np.ndarray], targets: list[np.ndarray]) -> float:
        return np.mean(
            [self.loss(target, row >> self) for row, target in zip(data, targets)]
        )

    def load_backup(self):
        self.layers = self.backup_layers
        self.solver.layers = self.backup_layers

    def save_backup(self):
        self.backup_layers = [copy(layer) for layer in self.layers]

    def dump(self) -> str:
        self.backup_layers = None
        return json.dumps(self, cls=NeuralNetworkSerializer)

    def to_file(self, filename: str = None, *, folder: str = "./nndumps"):
        if not filename:
            dims = (
                "_".join(f"{layer.input_dimension}" for layer in self.layers[::2])
                + f"_{self.layers[-1].output_dimension}"
            )
            layers_repr = "_".join(layer.type.name for layer in self.layers)
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.__class__.__name__}_{dims}_{layers_repr}_{current_datetime}.nnsf"

        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        path = path / filename
        with open(path, "w") as file:
            file.write(self.dump())

    @classmethod
    def from_str(cls, _str: str) -> "Perceptron":
        return json.loads(_str, cls=NeuralNetworkDeserializer)

    @classmethod
    def from_file(cls, filename: str, *, folder: str = "./nndumps") -> "Perceptron":
        path = Path(folder) / filename
        with open(path, "r") as file:
            return cls.from_str(file.read())

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
        activations: list[Literal["sigmoid", "heaviside", "softmax", "relu"]] = None,
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
        activations: Literal["sigmoid", "heaviside", "softmax", "relu"] = None,
        args_list: list[tuple[Any]] = None,
    ) -> "Perceptron": ...

    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        *,
        loss: Literal["cross_entropy", "squared_error"] = "squared_error",
        solver: Literal["gradient_descent"] = "gradient_descent",
        activations: Differentiable | list[Differentiable] = None,
        args_list: tuple[Any] | list[tuple[Any]] = None,
    ) -> "Perceptron":
        dims = list(dims)

        if any(not isinstance(size, int) for size in dims):
            raise TypeError("All sizes in dimensions should be python integers")
        if len(dims) < 2:
            raise ValueError("Dimensions should have 2 or more elements")

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

    def check_target(point: np.ndarray[float]) -> np.ndarray[float]:
        if np.linalg.norm(point) < 0.65:
            return [0, 0, 1]
        if np.linalg.norm(point - [0.5, 1]) < 0.45:
            return [0, 1, 0]
        return [1, 0, 0]

    target = np.array([check_target(point) for point in data])

    perceptron = Perceptron.from_dimensions(
        (2, 10, 10, 3),
        loss="cross_entropy",
        activations=["relu", "relu", "relu"],
    )
    perceptron.learn(data, target, learn_rate=0.4, while_not_learned=True)
