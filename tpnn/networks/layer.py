from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal, override
from scipy import signal

import numpy as np
import json


from tpnn.core.pipeline import Pipeable
from .activation import Differentiable, ActivationBuilder, AvailableActivations


class _Type(Enum):
    BASE = "base"
    FC = "fully_connected"
    ACT = "activation"
    CONV = "convolution"
    POOL = "pooling"
    DO = "dropout"
    BN = "batch_normalization"
    FLAT = "flatten"


class LayerSerializer(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, LayerBase):
            layer = asdict(
                o,
            )
            del layer["input"]
            del layer["output"]
            if isinstance(o, ACTLayer):
                layer["activation"] = o.activation
            return {"type": o.type.value, "layer_body": layer}

        if isinstance(o, Differentiable):
            func = o.func
            args = (
                [
                    cell.cell_contents
                    for cell in func.__closure__
                    if not callable(cell.cell_contents)
                ]
                if func.__closure__
                else []
            )
            func = func.__name__
            return {
                "func": func[: func.rfind("_activation")],
                "args": args,
            }

        if isinstance(o, np.ndarray):
            return {"array": o.tolist()}

        print(o)

        return super().default(o)


class LayerDeserializer(json.JSONDecoder):
    def __init__(self, *args, object_hook=None, **kwargs):
        super().__init__(
            *args,
            object_hook=self.object_hook if object_hook is None else self.object_hook,
            **kwargs,
        )

    def object_hook(self, _dict):
        match _dict:
            case {"type": _type, "layer_body": body}:
                cls = LayerBase.from_type(_Type(_type))
                return cls(**body)
            case {"func": func, "args": args}:
                return ActivationBuilder().build(func, args=args)
            case {"array": array}:
                return np.array(array)
            case _:
                return _dict


# TODO inherit LayerBase from Pipeable and BackPipeable
#      input >> layer      --- forward
#      output << layer     --- backward


@dataclass
class LayerBase(Pipeable[np.ndarray[np.number], np.ndarray[np.number]]):
    type: ClassVar[_Type] = field(default=_Type.BASE, init=False)
    input_dimension: int | tuple[int]
    output_dimension: int | tuple[int]
    input: np.ndarray = field(default=None, init=False, repr=False)
    output: np.ndarray = field(default=None, init=False, repr=False)

    _registered: ClassVar[dict[_Type, "LayerBase"]] = field(
        default={}, init=False, repr=False
    )

    def dump(self) -> str:
        return json.dumps(self, cls=LayerSerializer)

    def to_file(self, filename: str, *, folder: str = "./nndumps"):
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        path = path / filename
        with open(path, "w") as file:
            file.write(self.dump())

    @classmethod
    def from_str(cls, _str: str) -> "LayerBase":
        return json.loads(_str, cls=LayerDeserializer)

    @classmethod
    def from_file(cls, filename: str, *, folder: str = "./nndumps") -> "LayerBase":
        path = Path(folder) / filename
        with open(path, "r") as file:
            return cls.from_str(file.read())

    @classmethod
    def from_type(cls, _type: _Type) -> "LayerBase":
        return cls._registered.get(_type, None)


@dataclass
class ACTLayer(LayerBase):
    type: ClassVar[_Type] = field(default=_Type.ACT, init=False)
    output_dimension: int | tuple[int] = field(default=None, init=False)
    activation: Differentiable | AvailableActivations = field(
        default="sigmoid", kw_only=True
    )
    args: tuple[Any] = field(default=(), kw_only=True, repr=False, compare=False)

    @override
    def __call__(self, _input: np.ndarray) -> np.ndarray:
        self.input = _input
        self.output = self.activation(_input)
        return self.output

    def __post_init__(self):
        self.output_dimension = self.input_dimension
        if isinstance(self.activation, str):
            self.activation = ActivationBuilder().build(self.activation, args=self.args)
            del self.args


@dataclass
class FCLayer(LayerBase):
    type: ClassVar[_Type] = field(default=_Type.FC, init=False)
    weights: np.ndarray = field(default=None, kw_only=True, repr=False)
    biases: np.ndarray = field(default=None, kw_only=True, repr=False)

    @override
    def __call__(self, _input: np.ndarray) -> np.ndarray:
        self.input = _input
        self.output = self.input @ self.weights + self.biases
        return self.output

    def __post_init__(self):
        if self.weights is None:
            self.weights = (
                np.random.random((self.input_dimension, self.output_dimension)) * 2 - 1
            )
        if self.biases is None:
            self.biases = np.random.random((1, self.output_dimension)) * 2 - 1


@dataclass
class FLATLayer(LayerBase):
    type: ClassVar[_Type] = field(default=_Type.FLAT, init=False)

    @override
    def __call__(
        self, _input: np.ndarray[np.number, Any]
    ) -> np.ndarray[np.number, Any]:
        self.input = _input
        self.output = np.reshape(_input, self.output_dimension)
        return self.output


@dataclass
class CONVLayer(LayerBase):
    type: ClassVar[_Type] = field(default=_Type.CONV, init=False)
    kernel_size: int = field(init=True)
    depth: int = field(init=True)
    kernels: np.ndarray[float] = field(
        default=None, init=False, kw_only=True, repr=False
    )
    biases: np.ndarray[float] = field(
        default=None, init=False, kw_only=True, repr=False
    )
    padding: int = field(default=0, kw_only=True)

    @override
    def __call__(self, _input: np.ndarray) -> np.ndarray:
        self.input = _input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input[j], self.kernels[i, j], "valid"
                )
        return self.output

    def __post_init__(self):
        input_depth, input_height, input_width = self.input_dimension
        self.input_depth = input_depth
        self.output_shape = (
            self.depth,
            input_height - self.kernel_size + 1,
            input_width - self.kernel_size + 1,
        )
        self.kernels_shape = (
            self.depth,
            input_depth,
            self.kernel_size,
            self.kernel_size,
        )
        if self.kernels is None:
            self.kernels = np.random.randn(*self.kernels_shape)
        if self.biases is None:
            self.biases = np.random.randn(*self.output_shape)


@dataclass
class POOLLayer(LayerBase):
    type: ClassVar[_Type] = field(default=_Type.POOL, init=False)
    output_dimension: tuple[int] = field(default=None, init=False)
    pool_size: int
    pool_type: Literal["max", "mean"] = field(default="max", kw_only=True)
    stride: int = field(default=0, kw_only=True)

    @override
    def __call__(
        self, _input: np.ndarray[np.number, Any]
    ) -> np.ndarray[np.number, Any]:
        self.input = _input
        depth, height, width = _input.shape
        pooled_height = height // self.pool_size
        pooled_width = width // self.pool_size
        self.output = np.zeros(self.output_dimension)

        for depth_iter in range(depth):
            for pool_y in range(pooled_height):
                for pool_x in range(pooled_width):
                    y_start = pool_y * self.pool_size
                    y_end = y_start + self.pool_size
                    x_start = pool_x * self.pool_size
                    x_end = x_start + self.pool_size
                    window = _input[depth_iter, y_start:y_end, x_start:x_end]
                    self.output[depth_iter, pool_y, pool_x] = self.pool_func(window)

        return self.output

    def __post_init__(self):
        self.pool_func = np.mean if self.pool_type == "mean" else np.max
        depth, height, width = self.input_dimension
        pooled_height = height // self.pool_size
        pooled_width = width // self.pool_size
        self.output_dimension = (depth, pooled_height, pooled_width)


for cls in (LayerBase, ACTLayer, FCLayer, CONVLayer, FLATLayer, POOLLayer):
    LayerBase._registered[cls.type] = cls

if __name__ == "__main__":
    print(LayerBase._registered)
    fc1 = FCLayer(3, 5)
    act1 = ACTLayer(5, 5)
    fc2 = FCLayer(5, 3)
    act2 = ACTLayer(3, 3)

    print(ACTLayer.from_str(act2.dump()))
    print(FCLayer.from_str(fc2.dump()))

    data = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 0, 1],
        ]
    )
    print(data[0] >> fc1)
    print(data[0] >> act1)
    print(data >> fc1)
    print(data >> act1)
