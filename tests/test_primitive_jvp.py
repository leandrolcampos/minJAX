import itertools

import mpmath as mp  # type: ignore
import numpy as np
import pytest

from minjax import dtypes, lax, utils
from minjax.interpreters import ad

DEFAULT_SHAPES = ((), (2,), (2, 3))
DEFAULT_FLOATING_DTYPES = (np.float32, np.float64)


def _create_random_arrays(num, scale, shape, dtype, seed=42):
    args = []

    rng = np.random.default_rng(seed=seed)

    for _ in range(num):
        x = rng.normal(scale=scale, size=shape).astype(dtype)
        x = dtype(x.item()) if x.shape == () else x
        args.append(x)

    return args


def _make_fn(f, args_left, args_right, params):
    return lambda _x: f(*args_left, _x, *args_right, **params)


def _test_primal(
    minjax_impl,
    args,
    params=None,
    rtol=1e-07,
    atol=0,
):
    params = {} if params is None else params

    n_args = len(args)
    for i in range(n_args):
        x = args[i]
        ones = utils.ones_like(x)

        args_left = args[:i]
        args_right = args[(i + 1) :]

        minjax_fn = _make_fn(minjax_impl, args_left, args_right, params)
        out, _ = (np.asarray(out) for out in ad.jvp_v1(minjax_fn, (x,), (ones,)))

        expected = minjax_fn(x)

        assert out.shape == expected.shape
        assert out.dtype == expected.dtype

        np.testing.assert_allclose(out, expected, rtol=rtol, atol=atol)


def _test_tangent(
    minjax_impl,
    mpmath_impl,
    dtype,
    args,
    params=None,
    rtol=1e-07,
    atol=0,
):
    params = {} if params is None else params

    n_args = len(args)
    for i in range(n_args):
        x = args[i]
        ones = utils.ones_like(x)

        args_left = args[:i]
        args_right = args[(i + 1) :]

        minjax_fn = _make_fn(minjax_impl, args_left, args_right, params)
        _, out = (np.asarray(out) for out in ad.jvp_v1(minjax_fn, (x,), (ones,)))

        mpmath_fn = _make_fn(mpmath_impl, args_left, args_right, params)
        with mp.workdps(25):  # Set the decimal precision of mpmath.
            expected = dtype(mp.diff(mpmath_fn, x))

        assert out.shape == expected.shape
        assert out.dtype == expected.dtype

        if rtol == 0.0 and atol == 0.0:
            np.testing.assert_equal(out, expected)
        else:
            np.testing.assert_allclose(out, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape, dtype",
    itertools.product(DEFAULT_SHAPES, DEFAULT_FLOATING_DTYPES),
)
def test_neg(shape, dtype):
    def mpmath_impl(x):
        return -x

    args = _create_random_arrays(1, 100.0, shape, dtype)

    _test_primal(lax.neg, args)
    _test_tangent(lax.neg, mpmath_impl, dtype, args)


@pytest.mark.parametrize(
    "shape, dtype",
    itertools.product(DEFAULT_SHAPES, DEFAULT_FLOATING_DTYPES),
)
def test_add(shape, dtype):
    def mpmath_impl(x, y):
        return x + y

    args = _create_random_arrays(2, 100.0, shape, dtype)

    _test_primal(lax.add, args)
    _test_tangent(lax.add, mpmath_impl, dtype, args)


@pytest.mark.parametrize(
    "shape, dtype",
    itertools.product(DEFAULT_SHAPES, DEFAULT_FLOATING_DTYPES),
)
def test_mul(shape, dtype):
    def mpmath_impl(x, y):
        return x * y

    args = _create_random_arrays(2, 100.0, shape, dtype)

    _test_primal(lax.mul, args)
    _test_tangent(lax.mul, mpmath_impl, dtype, args)


@pytest.mark.parametrize(
    "shape, dtype",
    itertools.product(DEFAULT_SHAPES, DEFAULT_FLOATING_DTYPES),
)
def test_greater(shape, dtype):
    def mpmath_impl(x, y):
        return x + y

    args = _create_random_arrays(2, 100.0, shape, dtype)

    def _dtype(x):
        return np.zeros_like(x, dtype=dtypes.float0)

    _test_primal(lax.greater, args)
    _test_tangent(lax.greater, mpmath_impl, _dtype, args, rtol=0.0)


@pytest.mark.parametrize(
    "shape, dtype",
    itertools.product(DEFAULT_SHAPES, DEFAULT_FLOATING_DTYPES),
)
def test_less(shape, dtype):
    def mpmath_impl(x, y):
        return x + y

    args = _create_random_arrays(2, 100.0, shape, dtype)

    def _dtype(x):
        return np.zeros_like(x, dtype=dtypes.float0)

    _test_primal(lax.less, args)
    _test_tangent(lax.less, mpmath_impl, _dtype, args, rtol=0.0)


@pytest.mark.parametrize(
    "shape, dtype",
    itertools.product(DEFAULT_SHAPES, DEFAULT_FLOATING_DTYPES),
)
def test_sin(shape, dtype):
    def mpmath_impl(x):
        return np.frompyfunc(mp.sin, 1, 1)(x)

    args = _create_random_arrays(1, 100.0, shape, dtype)

    _test_primal(lax.sin, args)
    _test_tangent(lax.sin, mpmath_impl, dtype, args, rtol=1e-06)


@pytest.mark.parametrize(
    "shape, dtype",
    itertools.product(DEFAULT_SHAPES, DEFAULT_FLOATING_DTYPES),
)
def test_cos(shape, dtype):
    def mpmath_impl(x):
        return np.frompyfunc(mp.cos, 1, 1)(x)

    args = _create_random_arrays(1, 100.0, shape, dtype)

    _test_primal(lax.cos, args)
    _test_tangent(lax.cos, mpmath_impl, dtype, args, rtol=1e-06)


@pytest.mark.parametrize(
    "shape, dtype, broadcast_shape, axes",
    [
        ((), np.float32, (1, 5), None),
        ((2, 1), np.float32, (2, 5), None),
        ((2, 1), np.float32, (2, 5, 3), (2,)),
        ((2, 3), np.float32, (2, 5, 3), (1,)),
        ((2, 1), np.float64, (2, 5), None),
    ],
)
def test_broadcast(shape, dtype, broadcast_shape, axes):
    def mpmath_impl(x, shape, axes):
        if axes is not None:
            for axis in sorted(axes):
                x = np.expand_dims(x, axis)
        return np.broadcast_to(x, shape)

    args = _create_random_arrays(1, 100.0, shape, dtype)
    params = {"shape": broadcast_shape, "axes": axes}

    _test_primal(lax.broadcast, args, params)
    _test_tangent(lax.broadcast, mpmath_impl, dtype, args, params)


@pytest.mark.parametrize(
    "shape, dtype, perm",
    [
        ((), np.float32, None),
        ((2,), np.float32, None),
        ((2, 3, 4), np.float32, (0, 1, 2)),
        ((2, 3, 4), np.float32, (0, 2, 1)),
        ((2, 3, 4), np.float32, (2, 0, 1)),
        ((2, 3, 4), np.float32, (2, 1, 0)),
        ((2, 3, 4), np.float32, None),
        ((2, 3, 4), np.float64, None),
    ],
)
def test_transpose(shape, dtype, perm):
    def mpmath_impl(x, perm):
        return np.transpose(x, axes=perm)

    args = _create_random_arrays(1, 100.0, shape, dtype)
    params = {"perm": perm}

    _test_primal(lax.transpose, args, params)
    _test_tangent(lax.transpose, mpmath_impl, dtype, args, params)


@pytest.mark.parametrize(
    "shape, dtype, axis",
    [
        ((), np.float32, None),
        ((2,), np.float32, (0,)),
        ((2, 3), np.float32, (0,)),
        ((2, 3), np.float32, (0, 1)),
        ((2, 3), np.float32, None),
        ((2, 3), np.float64, None),
    ],
)
def test_reduce_sum(shape, dtype, axis):
    def mpmath_impl(x, axis):
        return np.sum(x, axis=axis)

    args = _create_random_arrays(1, 100.0, shape, dtype)
    params = {"axis": axis}

    _test_primal(lax.reduce_sum, args, params)
    _test_tangent(lax.reduce_sum, mpmath_impl, dtype, args, params)
