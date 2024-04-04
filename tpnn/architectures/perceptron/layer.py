from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import numpy as np
import json

from typing import overload

from .activation import Activation, sigmoid, ActivationBuilder


class LayerSerializer(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Layer):
            return asdict(o)

        if isinstance(o, np.ndarray):
            return {"array": o.tolist()}

        if callable(o):
            args = (
                [cell.cell_contents for cell in o.__closure__] if o.__closure__ else []
            )
            return {
                "func": o.__name__,
                "args": args,
            }

        return super().default(o)


class LayerDeserializer(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, object_hook=self.object_hook, **kwargs)

    def object_hook(self, _dict):
        match _dict:
            case {"func": func, "args": args}:
                return ActivationBuilder().build(func, args=args)
            case {"array": array}:
                return np.array(array)
            case _:
                return _dict


@dataclass
class Layer:
    input_dimension: int
    output_dimension: int
    activation: Activation = field(default=sigmoid, kw_only=True)
    weights: np.ndarray = field(default=None, kw_only=True)
    biases: np.ndarray = field(default=None, kw_only=True)

    def __call__(self, _input: np.ndarray) -> np.ndarray:
        return self.activation(_input @ self.weights + self.biases)

    def dump(self) -> str:
        return json.dumps(self, cls=LayerSerializer)

    def to_file(self, filename: str, *, folder: str = "./nndumps"):
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        path = path / filename
        with open(path, "w") as file:
            file.write(self.dump())

    @classmethod
    def from_str(cls, _str: str) -> "Layer":
        return cls(**json.loads(_str, cls=LayerDeserializer))

    @classmethod
    def from_file(cls, filename: str, *, folder: str = "./nndumps") -> "Layer":
        path = Path(folder) / filename
        with open(path, "r") as file:
            return cls.from_str(file.read())

    def __post_init__(self):
        if self.weights is None:
            self.weights = (
                np.random.random((self.input_dimension, self.output_dimension)) * 2 - 1
            )
        if self.biases is None:
            self.biases = np.random.random(self.output_dimension) * 2 - 1
