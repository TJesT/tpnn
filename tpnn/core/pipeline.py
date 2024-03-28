from typing import TypeVar, Callable, get_args, Union, overload
from abc import ABC, abstractmethod


Input = TypeVar("Input")
Output = TypeVar("Output")
OtherInput = TypeVar("OtherInput")
OtherOutput = TypeVar("OtherOutput")


def _validate_IO(
    left: Union["Pipeable", "Pipeline"],
    right: Union["Pipeable", "Pipeline"],
) -> tuple[type, type]:
    left_in_type, left_out_type = left._io_types
    right_in_type, right_out_type = right._io_types

    if f"{left_out_type}" != f"{right_in_type}":
        raise TypeError(
            f"Output type `{left_out_type}` of {left} "
            f"mismatches input type `{right_in_type}` of {right}."
        )

    return left_in_type, right_out_type


class Pipeline[Input, Output]:
    def __init__(self, *nodes) -> None:
        self.nodes = nodes

    def __call__(self, _input: Input) -> Output:
        result = self.nodes[0](_input)

        for node in self.nodes[1:]:
            result = node(result)

        return result

    @overload
    def __rrshift__(self, _input: Input) -> Output: ...

    @overload
    def __rrshift__(
        self, _input: "Pipeable[OtherInput, Input]"
    ) -> "Pipeline[OtherInput, Output]": ...

    @overload
    def __rrshift__(
        self, _input: "Pipeline[OtherInput, Input]"
    ) -> "Pipeline[OtherInput, Output]": ...

    def __rrshift__(self, _input):
        match _input:
            case Pipeable():
                new_in, new_out = _validate_IO(_input, self)
                return Pipeline[new_in, new_out](_input, self)

            case Pipeline():
                new_in, new_out = _validate_IO(_input, self)
                return Pipeline[new_in, new_out](*_input.nodes, self)

            case _:
                return self(_input)

    @property
    def _io_types(self) -> tuple[type, type]:
        return get_args(self.__orig_class__)


class Pipeable[Input, Output](ABC):
    @abstractmethod
    def __call__(self, _input: Input) -> Output: ...

    @overload
    def __rrshift__(self, _input: Input) -> Output: ...

    @overload
    def __rrshift__(
        self, _input: "Pipeable[OtherInput, Input]"
    ) -> Pipeline[OtherInput, Output]: ...

    @overload
    def __rrshift__(
        self, _input: Pipeline[OtherInput, Input]
    ) -> Pipeline[OtherInput, Output]: ...

    def __rrshift__(self, _input):
        match _input:
            case Pipeable():
                new_in, new_out = _validate_IO(_input, self)
                return Pipeline[new_in, new_out](_input, self)
            case Pipeline():
                new_in, new_out = _validate_IO(_input, self)
                return Pipeline[new_in, new_out](*_input.nodes, self)
            case _:
                return self(_input)

    @overload
    def __rshift__(
        self, _input: "Pipeable[Output, OtherOutput]"
    ) -> Pipeline[Input, OtherOutput]: ...

    @overload
    def __rshift__(
        self, _input: Pipeline[Output, OtherOutput]
    ) -> Pipeline[Input, OtherOutput]: ...

    def __rshift__(self, _input):
        match _input:
            case Pipeable():
                new_in, new_out = _validate_IO(self, _input)
                return Pipeline[new_in, new_out](self, _input)
            case Pipeline():
                new_in, new_out = _validate_IO(self, _input)
                return Pipeline[new_in, new_out](self, *_input.nodes)

    @property
    def _io_types(self) -> tuple[type, type]:
        return get_args(self.__orig_bases__[0])
