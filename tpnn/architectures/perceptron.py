import numpy as np
from dataclasses import dataclass
from typing import Iterable, overload
from .base.layer import Layer


@dataclass
class Perceptron:
    layers: list[Layer]

    @overload
    def predict(self, _input: list[np.number]) -> np.ndarray: ...
    @overload
    def predict(self, _input: Iterable[np.number]) -> np.ndarray: ...

    def predict(self, _input: np.ndarray) -> np.ndarray:
        if not isinstance(_input, np.ndarray):
            _input = np.array(_input)

        if (ishape := _input.shape[0]) != (lshape := self.layers[0].input_dimension):
            raise ValueError(
                f"Shape of input={ishape} doesn't fit first layer's shape={lshape}"
            )

        curr = _input
        for i, layer in enumerate(self.layers, 1):
            print(f"Passing {curr} through layer #{i}")
            curr = layer(curr)

        return curr

    @classmethod
    def from_dimensions(cls, dims: Iterable[int]) -> "Perceptron":
        dims = list(dims)

        if any(not isinstance(size, int) for size in dims):
            raise TypeError("All sizes in dimensions should be python integers")
        if len(dims) < 2:
            raise ValueError("Dimenshions should have 2 or more elements")

        return Perceptron(
            layers=[Layer(idim, odim) for idim, odim in zip(dims, dims[1:])]
        )
