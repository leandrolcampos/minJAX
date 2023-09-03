"""Automatic differentiation."""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from minjax import core, dtypes, utils
from minjax.typing import Array, ArrayLike

__all__ = [
    "jvp_v1",
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
        self, trace: core.Trace, primal: ArrayLike, tangent: ArrayLike
    ) -> None:
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def aval(self) -> core.ShapedArray:
        return core.get_aval(self.primal)


class JVPTrace(core.Trace[JVPTracer]):
    def pure(self, x: ArrayLike) -> JVPTracer:
        return JVPTracer(trace=self, primal=x, tangent=utils.zeros_like(x))

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


def jvp_v1(f: Callable, primals: Sequence[ArrayLike], tangents: Sequence[ArrayLike]):
    with core.new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in: list[JVPTracer] = []

        for p, t in utils.safe_zip(primals, tangents):
            primal_dtype = dtypes.get_dtype(p)
            if np.issubdtype(primal_dtype, np.floating):
                tangent_dtype = primal_dtype
            else:
                tangent_dtype = dtypes.float0

            primal = np.asarray(p, dtype=primal_dtype)
            tangent = np.broadcast_to(
                np.asarray(t, dtype=tangent_dtype), shape=primal.shape
            )

            tracers_in.append(JVPTracer(trace, primal, tangent))

        out = f(*tracers_in)
        tracer_out = trace.full_raise(out)
        primal_out, tangent_out = tracer_out.primal, tracer_out.tangent

    return primal_out, tangent_out
