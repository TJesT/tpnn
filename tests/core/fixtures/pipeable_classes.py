from typing import override
from tpnn.core import Pipeable
from pytest import fixture


class A(Pipeable[int, str]):
    @override
    def __call__(self, _input: int) -> str:
        return str(_input)


class B(Pipeable[str, list[str]]):
    @override
    def __call__(self, _input: str) -> list:
        return list(_input)


class C(Pipeable[list[str], int]):
    @override
    def __call__(self, _input: list[str]) -> int:
        return int("".join(_input))


@fixture
def a():
    return A()


@fixture
def b():
    return B()


@fixture
def c():
    return C()
