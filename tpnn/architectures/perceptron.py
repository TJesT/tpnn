import numpy as np
from dataclasses import dataclass
from typing import Any, Iterable, overload, override, Literal

from .base.activation import Activation, ActivationBuilder
from .base.layer import FCLayer, ACTLayer
from tpnn.core import Pipeable


@dataclass
class Perceptron(Pipeable[np.ndarray[np.number], np.ndarray[np.number]]):
    layers: list[FCLayer | ACTLayer]

    @override
    def __call__(self, _input: np.ndarray) -> np.ndarray:
        if (ishape := _input.shape[0]) != (lshape := self.layers[0].input_dimension):
            raise ValueError(
                f"Shape of input={ishape} doesn't fit first layer's shape={lshape}"
            )

        curr = _input
        for i, layer in enumerate(self.layers, 1):
            print(f"Passing {curr} through layer #{i}({layer.__class__.type.name})")
            curr = curr >> layer

        return curr

    @overload
    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        *,
        activations: list[Activation] = None,
    ) -> "Perceptron": ...

    @overload
    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        *,
        activations: list[Literal["sigmoid", "heaviside"]] = None,
        args_list: list[tuple[Any]] = None,
    ) -> "Perceptron": ...

    @overload
    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        *,
        activations: Activation = None,
    ) -> "Perceptron": ...

    @overload
    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        *,
        activations: Literal["sigmoid", "heaviside"] = None,
        args_list: list[tuple[Any]] = None,
    ) -> "Perceptron": ...

    @classmethod
    def from_dimensions(
        cls,
        dims: Iterable[int],
        *,
        activations: Activation | list[Activation] = None,
        args_list: tuple[Any] | list[tuple[Any]] = None,
    ) -> "Perceptron":
        dims = list(dims)

        if any(not isinstance(size, int) for size in dims):
            raise TypeError("All sizes in dimensions should be python integers")
        if len(dims) < 2:
            raise ValueError("Dimensions should have 2 or more elements")

        if not isinstance(activations, list):
            activations = [activations] * (len(dims) - 1)

        if len(activations) != len(dims) - 1:
            raise ValueError(
                "List of Activations should have `len(Dimensions) - 1` elements. "
                f"But you have {len(activations) = } instead of len(activation) = {len(dims)-1}"
            )

        activations = [
            ActivationBuilder().build(act, args=args) if isinstance(act, str) else act
            for act, args in zip(activations, args_list)
        ]

        return cls(
            layers=[
                (
                    lcls(idim, odim)
                    if lcls is FCLayer
                    else lcls(odim, odim, activation=activation)
                )
                for idim, odim, activation in zip(dims, dims[1:], activations)
                for lcls in [FCLayer, ACTLayer]
            ]
        )


if __name__ == "__main__":
    perceptron = Perceptron.from_dimensions(
        (3, 5, 3),
        activations=["heaviside"],
        args_list=[(0.5,), (0,)],
    )
    print(np.array([0, 1, 0]) >> perceptron)
