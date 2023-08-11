"""Tests evaluation of primitives."""

import itertools

import numpy as np
import pytest

from minjax import lax

DEFAULT_SHAPES = [(), (2,), (2, 3)]
DEFAULT_INTEGER_DTYPES = [np.int32, np.int64]
DEFAULT_FLOATING_DTYPES = [np.float32, np.float64]
DEFAULT_DTYPES = DEFAULT_INTEGER_DTYPES + DEFAULT_FLOATING_DTYPES


def _create_random_arrays(num, scale, shape, dtype):
    args = []

    for _ in range(num):
        x = np.random.normal(scale=scale, size=shape).astype(dtype)
        x = x.item() if x.shape == () else x
        args.append(x)

    return args


def _test_primitive_eval(minjax_op, numpy_impl, args, params=None, rtol=1e-07, atol=0):
    params = {} if params is None else params

    out = np.asarray(minjax_op(*args, **params))
    expected = numpy_impl(*args, **params)

    assert out.shape == expected.shape
    assert out.dtype == expected.dtype

    np.testing.assert_allclose(out, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape, dtype", itertools.product(DEFAULT_SHAPES, DEFAULT_DTYPES)
)
def test_neg(shape, dtype):
    args = _create_random_arrays(1, 100.0, shape, dtype)

    rtol = 1e-07 if np.issubdtype(dtype, np.floating) else 0.0
    atol = 0.0

    _test_primitive_eval(lax.neg, np.negative, args, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape, dtype", itertools.product(DEFAULT_SHAPES, DEFAULT_DTYPES)
)
def test_add(shape, dtype):
    args = _create_random_arrays(2, 100.0, shape, dtype)

    rtol = 1e-07 if np.issubdtype(dtype, np.floating) else 0.0
    atol = 0.0

    _test_primitive_eval(lax.add, np.add, args, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape, dtype", itertools.product(DEFAULT_SHAPES, DEFAULT_DTYPES)
)
def test_mul(shape, dtype):
    args = _create_random_arrays(2, 100.0, shape, dtype)

    rtol = 1e-07 if np.issubdtype(dtype, np.floating) else 0.0
    atol = 0.0

    _test_primitive_eval(lax.mul, np.multiply, args, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape, dtype", itertools.product(DEFAULT_SHAPES, DEFAULT_DTYPES)
)
def test_greater(shape, dtype):
    args = _create_random_arrays(2, 100.0, shape, dtype)

    rtol = 1e-07 if np.issubdtype(dtype, np.floating) else 0.0
    atol = 0.0

    _test_primitive_eval(lax.greater, np.greater, args, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape, dtype", itertools.product(DEFAULT_SHAPES, DEFAULT_DTYPES)
)
def test_less(shape, dtype):
    args = _create_random_arrays(2, 100.0, shape, dtype)

    rtol = 1e-07 if np.issubdtype(dtype, np.floating) else 0.0
    atol = 0.0

    _test_primitive_eval(lax.less, np.less, args, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "shape, dtype", itertools.product(DEFAULT_SHAPES, DEFAULT_FLOATING_DTYPES)
)
def test_sin(shape, dtype):
    args = _create_random_arrays(1, 100.0, shape, dtype)

    _test_primitive_eval(lax.sin, np.sin, args, rtol=1e-07, atol=0.0)


@pytest.mark.parametrize(
    "shape, dtype", itertools.product(DEFAULT_SHAPES, DEFAULT_FLOATING_DTYPES)
)
def test_cos(shape, dtype):
    args = _create_random_arrays(1, 100.0, shape, dtype)

    _test_primitive_eval(lax.cos, np.cos, args, rtol=1e-07, atol=0.0)


@pytest.mark.parametrize(
    "shape, dtype, broadcast_shape, axes",
    [
        ((2, 1), np.int32, (2, 5), None),
        ((2, 1), np.int64, (2, 5), None),
        ((2, 1), np.float64, (2, 5), None),
        ((), np.float32, (1, 5), None),
        ((2, 1), np.float32, (2, 5), None),
        ((2, 1), np.float32, (2, 5, 3), (2,)),
        ((2, 3), np.float32, (2, 5, 3), (1,)),
    ],
)
def test_broadcast(shape, dtype, broadcast_shape, axes):
    args = _create_random_arrays(1, 100.0, shape, dtype)
    params = {"shape": broadcast_shape, "axes": axes}

    def numpy_impl(x, shape, axes):
        if axes is not None:
            for axis in sorted(axes):
                x = np.expand_dims(x, axis)
        return np.broadcast_to(x, shape)

    _test_primitive_eval(lax.broadcast, numpy_impl, args, params, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "shape, dtype, perm",
    [
        ((2, 3, 4), np.int32, None),
        ((2, 3, 4), np.int64, None),
        ((2, 3, 4), np.float64, None),
        ((), np.float32, None),
        ((2,), np.float32, None),
        ((2, 3, 4), np.float32, (0, 1, 2)),
        ((2, 3, 4), np.float32, (0, 2, 1)),
        ((2, 3, 4), np.float32, (2, 0, 1)),
        ((2, 3, 4), np.float32, (2, 1, 0)),
        ((2, 3, 4), np.float32, None),
    ],
)
def test_transpose(shape, dtype, perm):
    args = _create_random_arrays(1, 100.0, shape, dtype)
    params = {"perm": perm}

    def numpy_impl(x, perm):
        return np.transpose(x, axes=perm)

    _test_primitive_eval(lax.transpose, numpy_impl, args, params, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "shape, dtype, axis",
    [
        ((2, 3), np.int32, None),
        ((2, 3), np.int64, None),
        ((2, 3), np.float64, None),
        ((), np.float32, None),
        ((2,), np.float32, (0,)),
        ((2, 3), np.float32, (0,)),
        ((2, 3), np.float32, (0, 1)),
        ((2, 3), np.float32, None),
    ],
)
def test_reduce_sum(shape, dtype, axis):
    args = _create_random_arrays(1, 100.0, shape, dtype)
    params = {"axis": axis}

    rtol = 1e-07 if np.issubdtype(dtype, np.floating) else 0.0
    atol = 0.0

    _test_primitive_eval(lax.reduce_sum, np.sum, args, params, rtol=rtol, atol=atol)
