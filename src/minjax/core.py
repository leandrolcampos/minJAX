"""Core machinery."""

import contextlib
import operator as op
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, Generic, Protocol, TypeVar, Union

import numpy as np

from minjax import lax

__all__ = [
    "Primitive",
]

Array = Union[np.ndarray, "Tracer"]

NumpyArrayLike = (
    bool
    | int
    | float
    | np.bool_
    | np.int32
    | np.int64
    | np.float32
    | np.float64
    | np.ndarray
)

ArrayLike = Array | NumpyArrayLike

TracerType = TypeVar("TracerType", bound=Array)


class ImplRule(Protocol):
    def __call__(self, *args: ArrayLike, **params: Any) -> Array | Sequence[Array]:
        """Impl rule that runs a primitive operation using some interpreter."""
        ...


class Primitive:
    """Represents a primitive operation."""

    name: str
    multiple_results: bool = False
    impl_rule: None | ImplRule = None

    def __init__(self, name: str) -> None:
        self.name = name

    def bind(self, *args: ArrayLike, **params: Any) -> Array | Sequence[Array]:
        top_trace = find_top_trace(args)
        return self.bind_with_trace(top_trace, args, params)

    def bind_with_trace(
        self, trace: "Trace", args: Sequence[ArrayLike], params: dict[str, Any]
    ) -> Array | Sequence[Array]:
        tracers = map(trace.full_raise, args)
        out = trace.process_primitive(self, tracers, params)
        return map(full_lower, out) if self.multiple_results else full_lower(out)


# -------------------------------------- tracing ---------------------------------------


class Trace(Generic[TracerType], metaclass=ABCMeta):
    main: "MainTrace"

    def __init__(self, main: "MainTrace") -> None:
        self.main = main

    @abstractmethod
    def pure(self, x: ArrayLike) -> TracerType:
        ...

    @abstractmethod
    def lift(self, x: Array) -> TracerType:
        ...

    def full_raise(self, x: ArrayLike) -> TracerType:
        if not isinstance(x, Tracer):
            if not isinstance(x, NumpyArrayLike):
                error_message = (
                    f"Value {x} does not have a valid JAX type: `{type(x)}`."
                )
                raise TypeError(error_message)

            return self.pure(x)

        level = self.main.level
        if x._trace.main is self.main:
            return x

        if x._trace.main.level < level:
            return self.lift(x)

        if x._trace.main.level > level:
            error_message = f"Can't lift level {x._trace.main.level} to {level}."
            raise Exception(error_message)

        # (val._trace.main is not self.main) and (val._trace.level == level)
        error_message = f"Different traces at same level: {x._trace}, {self}."
        raise Exception(error_message)

    @abstractmethod
    def process_primitive(
        self,
        primitive: Primitive,
        tracers: Sequence[TracerType],
        params: dict[str, Any],
    ) -> Array | Sequence[Array]:
        ...


class Tracer(metaclass=ABCMeta):
    _trace: Trace
    __array_priority__: int = 1000

    @property
    @abstractmethod
    def aval(self) -> "ShapedArray":
        ...

    def full_lower(self) -> Array:
        return self

    def __neg__(self) -> Array:
        return self.aval._neg(self)

    def __add__(self, other: Array) -> Array:
        return self.aval._add(self, other)

    def __radd__(self, other: Array) -> Array:
        return self.aval._radd(self, other)

    def __mul__(self, other: Array) -> Array:
        return self.aval._mul(self, other)

    def __rmul__(self, other: Array) -> Array:
        return self.aval._rmul(self, other)

    def __gt__(self, other: Array) -> Array:
        return self.aval._gt(self, other)

    def __lt__(self, other: Array) -> Array:
        return self.aval._lt(self, other)

    def __bool__(self) -> bool:
        return self.aval._bool(self)

    def __nonzero__(self) -> bool:
        return self.aval._nonzero(self)

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self.aval, name)
        except AttributeError:
            error_message = f"{self.__class__.__name__} has no attribute {name}."
            raise AttributeError(error_message) from None


def find_top_trace(xs: Sequence[ArrayLike]) -> Trace:
    top_main = max(
        (x._trace.main for x in xs if isinstance(x, Tracer)),
        default=trace_stack[0],
        key=op.attrgetter("level"),
    )

    if dynamic_trace is not None and dynamic_trace.level > top_main.level:
        top_main = dynamic_trace

    return top_main.trace_type(top_main)


def full_lower(x: ArrayLike) -> ArrayLike:
    if isinstance(x, Tracer):
        return x.full_lower()

    return x


class EvalTrace(Trace[ArrayLike]):
    def pure(self, x: ArrayLike) -> ArrayLike:
        return x

    def lift(self, x: Array) -> ArrayLike:
        return x

    def process_primitive(
        self,
        primitive: Primitive,
        tracers: Sequence[ArrayLike],
        params: dict[str, Any],
    ) -> Array | Sequence[Array]:
        if primitive.impl_rule is None:
            error_message = f"Primitive `{primitive.name}` has no implementation rule."
            raise Exception(error_message)

        return primitive.impl_rule(*tracers, **params)


class MainTrace:
    def __init__(
        self, level: int, trace_type: type[Trace], global_data: Any = None
    ) -> None:
        self.level = level
        self.trace_type = trace_type
        self.global_data = global_data


trace_stack: list[MainTrace] = [MainTrace(0, EvalTrace)]
dynamic_trace: None | MainTrace = None


@contextlib.contextmanager
def new_main(trace_type: type[Trace], global_data: Any = None) -> Iterator[MainTrace]:
    level = len(trace_stack)
    main = MainTrace(level, trace_type, global_data)
    trace_stack.append(main)

    try:
        yield main
    finally:
        trace_stack.pop()


# ---------------------------------- abstract values -----------------------------------


class ShapedArray:
    array_abstraction_level: int = 1

    def __init__(self, shape: Sequence[int], dtype: np.dtype) -> None:
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @staticmethod
    def _neg(x: Array) -> Array:
        return lax.neg(x)

    @staticmethod
    def _add(x: Array, y: Array) -> Array:
        return lax.add(x, y)

    @staticmethod
    def _radd(x: Array, y: Array) -> Array:
        return lax.add(y, x)

    @staticmethod
    def _mul(x: Array, y: Array) -> Array:
        return lax.mul(x, y)

    @staticmethod
    def _rmul(x: Array, y: Array) -> Array:
        return lax.mul(y, x)

    @staticmethod
    def _gt(x: Array, y: Array) -> Array:
        return lax.greater(x, y)

    @staticmethod
    def _lt(x: Array, y: Array) -> Array:
        return lax.less(x, y)

    @staticmethod
    def _bool(x: Array) -> bool:
        del x
        error_message = "ShapedArray can't be unambiguously converted to bool."
        raise Exception(error_message)

    @staticmethod
    def _nonzero(x: Array) -> bool:
        del x
        error_message = "ShapedArray can't be unambiguously converted to bool."
        raise Exception(error_message)

    def str_short(self) -> str:
        return f"{self.dtype.name}[{','.join(str(d) for d in self.shape)}]"

    def __hash__(self) -> int:
        return hash((self.shape, self.dtype))

    def __eq__(self, other: "ShapedArray") -> bool:
        return (
            type(self) is type(other)
            and self.shape == other.shape
            and self.dtype == other.dtype
        )

    def __repr__(self) -> str:
        return f"ShapedArray(shape={self.shape}, dtype={self.dtype})"


class ConcreteArray(ShapedArray):
    array_abstraction_level = 2

    # TODO: replace `np.ndrray` with `DeviceArray`.
    def __init__(self, val: np.ndarray) -> None:
        self.val = val
        self.shape = val.shape
        self.dtype = val.dtype

    @staticmethod
    def _bool(x: Array) -> bool:
        return bool(x.aval.val)

    @staticmethod
    def _nonzero(x: Array) -> bool:
        return bool(x.aval.val)


def get_aval(x: ArrayLike) -> ShapedArray:
    if isinstance(x, Tracer):
        return x.aval

    if isinstance(x, NumpyArrayLike):
        return ConcreteArray(np.asarray(x))

    error_message = f"Value {x} does not have a valid JAX type: `{type(x)}`."
    raise TypeError(error_message)
