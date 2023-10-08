"""Automatic differentiation."""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from minjax import core, dtypes, pytrees, utils
from minjax.typing import Array, ArrayLike

__all__ = [
    "jvp",
    "register_differentiation",
]


# mypy implicitly sets this variable to true when type checking.
MYPY = False


if not MYPY:

    class JVPRule(Protocol):
        def __call__(
            self, primals: Sequence[Array], tangents: Sequence[Array], **params: Any
        ) -> tuple[Sequence[Array], Sequence[Array]]:
            """Performs the forward-mode automatic differentiation of a primitive."""
            ...

else:
    JVPRule = Any


_jvp_rules: dict[core.Primitive, JVPRule] = {}


def register_differentiation(primitive: core.Primitive, jvp_rule: JVPRule) -> None:
    _jvp_rules[primitive] = jvp_rule


class JVPTracer(core.Tracer):
    def __init__(
        self, trace: core.Trace, primal: ArrayLike, tangent: ArrayLike | None = None
    ) -> None:
        self._trace = trace

        primal_dtype = dtypes.get_dtype(primal)
        self.primal = np.asarray(primal, dtype=primal_dtype)

        if np.issubdtype(primal_dtype, np.floating):
            tangent_dtype = primal_dtype
        else:
            tangent_dtype = dtypes.float0

        if tangent is None:
            self.tangent = utils.zeros_like(self.primal, dtype=tangent_dtype)
        else:
            self.tangent = np.broadcast_to(
                np.asarray(tangent, dtype=tangent_dtype), shape=self.primal.shape
            )

    @property
    def aval(self) -> core.ShapedArray:
        return core.get_aval(self.primal)


class JVPTrace(core.Trace[JVPTracer]):
    def pure(self, x: ArrayLike) -> JVPTracer:
        return JVPTracer(trace=self, primal=x)

    lift = pure

    def process_primitive(
        self,
        primitive: core.Primitive,
        tracers: Sequence[JVPTracer],
        params: dict[str, Any],
    ) -> Sequence[JVPTracer]:
        primals_in, tangents_in = utils.unzip2([(t.primal, t.tangent) for t in tracers])

        jvp_rule = _jvp_rules.get(primitive)
        if jvp_rule is None:
            error_message = f"Primitive `{primitive.name}` has no JVP rule."
            raise NotImplementedError(error_message)

        primals_out, tangents_out = jvp_rule(primals_in, tangents_in, **params)
        return [
            JVPTracer(self, primal, tangent)
            for primal, tangent in utils.safe_zip(primals_out, tangents_out)
        ]


def jvp_flat(
    f: Callable, primals: Sequence[ArrayLike], tangents: Sequence[ArrayLike]
) -> tuple[Sequence[ArrayLike], Sequence[ArrayLike]]:
    with core.new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in = [
            JVPTracer(trace, primal, tangent)
            for primal, tangent in utils.safe_zip(primals, tangents)
        ]

        outs = f(*tracers_in)
        tracers_out = [trace.full_raise(out) for out in outs]
        primals_out, tangents_out = utils.unzip2(
            [(tracer_out.primal, tracer_out.tangent) for tracer_out in tracers_out]
        )

    return primals_out, tangents_out


def jvp(
    f: Callable, primals: Sequence[Any], tangents: Sequence[Any]
) -> tuple[Any, Any]:
    primals_flat, in_tree = pytrees.tree_flatten(primals)
    tangents_flat, in_tree2 = pytrees.tree_flatten(tangents)

    if in_tree != in_tree2:
        error_message = "The tree structures of `primals` and `tangents` must match."
        raise TypeError(error_message)

    f, out_tree = pytrees.flatten_fun(f, in_tree)
    primals_out_flat, tangents_out_flat = jvp_flat(f, primals_flat, tangents_flat)
    primals_out = pytrees.tree_unflatten(out_tree(), primals_out_flat)
    tangents_out = pytrees.tree_unflatten(out_tree(), tangents_out_flat)

    return primals_out, tangents_out
