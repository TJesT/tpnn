from abc import abstractmethod
import numpy as np
import json
from copy import copy
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator, override, SupportsIndex, Self


from tpnn.core.types import (
    Pipeable,
    WeightsGradients,
    BiasesGradients,
    BiasesGradient,
    WeightsGradient,
)
from .layer import LayerBase, FCLayer, ACTLayer, LayerDeserializer, LayerSerializer
from .loss import Loss, CEL, SEL
from .solver import Solver, GradientDescent


def batched(
    data: SupportsIndex, batch_size: int, length: int
) -> Generator[Any, None, None]:
    for start in range(0, length, batch_size):
        end = start + batch_size
        yield data[start:end]


class NeuralNetworkSerializer(LayerSerializer):
    def default(self, o: Any) -> Any:
        if isinstance(o, NeuralNetwork):
            return {
                "class": o.__class__.__name__,
                "layers": o.layers,
                "loss": self.default(o.loss),
                "solver": self.default(o.solver),
            }
        if isinstance(o, Solver):
            return {
                "type": o.type.value,
                "learn_rate": o.learn_rate,
                "momentum": o.momentum,
            }
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
                "class": class_name,
                "layers": layers,
                "loss": loss,
                "solver": solver,
            }:
                if class_name == "Perceptron":
                    from .perceptron import Perceptron

                    return Perceptron(layers=layers, loss=loss, solver=solver)
                return None
            case {"type": "loss", "name": loss_name}:
                if loss_name == "squared_error":
                    return SEL
                if loss_name == "cross_entropy":
                    return CEL
            case {"type": solver_type, "learn_rate": learn_rate, "momentum": momentum}:
                if solver_type == "gradient_descent":
                    return GradientDescent(learn_rate=learn_rate, momentum=momentum)
            case _:
                return LayerDeserializer.object_hook(self, _dict)


@dataclass
class NeuralNetwork(Pipeable[np.ndarray[np.number], np.ndarray[np.number]]):
    layers: list[LayerBase]
    loss: Loss
    solver: Solver
    best_cost: float = field(init=False, default=float("+inf"))
    backup_layers: list[FCLayer | ACTLayer] = field(init=False, default=None)

    @override
    def __call__(self, _input: np.ndarray[np.number]) -> np.ndarray[np.number]:
        if len(ishape := _input.shape) == 1 and ishape[0] != (
            lshape := self.layers[0].input_dimension
        ):
            raise ValueError(
                f"Shape of input={ishape[0]} doesn't fit first layer's shape={lshape}"
            )
        if len(ishape) == 2 and ishape[1] != (lshape := self.layers[0].input_dimension):
            raise ValueError(
                f"Shape of row in batch = {ishape[1]} doesn't fit first layer's shape={lshape}"
            )

        curr = _input
        for _, layer in enumerate(self.layers, 1):
            curr = curr >> layer

        return curr

    @abstractmethod
    def get_gradients(
        self, target: np.ndarray
    ) -> tuple[WeightsGradients, BiasesGradients]: ...

    @abstractmethod
    def update_weights(self, data: np.ndarray, target: np.ndarray) -> None: ...

    def cost(self, data: np.ndarray, targets: np.ndarray) -> float:
        return np.mean(self.loss(targets, data >> self))

    def learn(
        self,
        data: np.ndarray[np.ndarray],
        target: np.ndarray[np.ndarray],
        *,
        learn_rate: float = 0.1,
        momentum: float = 0.5,
        epochs: int = 20,
        batch_size: int = 1,
        while_not_learned: bool = False,
        # yield_self: bool = False,
    ) -> None:
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

        epochs_target = epochs + 1
        epochs += 1
        if isinstance(self.solver, GradientDescent):
            self.solver.learn_rate = learn_rate
            self.solver.momentum = momentum

        prev_cost = self.cost(data, target)
        print(f"{prev_cost = }")
        while while_not_learned or (epochs := epochs - 1):
            print(f"EPOCH #{epochs_target - epochs}: cost={self.cost(data, target)}")
            for data_batch, target_batch in zip(
                batched(data, batch_size, data.shape[0]),
                batched(target, batch_size, target.shape[0]),
            ):
                self.update_weights(data_batch, target_batch)
                curr_cost = self.cost(data, target)

                # if yield_self:
                #     yield (self, curr_cost)

                if abs(curr_cost) < abs(self.best_cost):
                    self.best_cost = curr_cost
                    self.save_backup()

                if abs(curr_cost) < 10**-10:
                    self.to_file()
                    while_not_learned = False
                    epochs = 1
                    break

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
    def from_str(cls, _str: str) -> Self:
        return json.loads(_str, cls=NeuralNetworkDeserializer)

    @classmethod
    def from_file(cls, filename: str, *, folder: str = "./nndumps") -> Self:
        path = Path(folder) / filename
        with open(path, "r") as file:
            return cls.from_str(file.read())
