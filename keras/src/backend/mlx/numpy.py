import builtins
import math

import mlx.core as mx
import numpy as np

from keras.src import tree
from keras.src.backend import config
from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.common.backend_utils import standardize_axis_for_numpy
from keras.src.backend.mlx.core import _flip
from keras.src.backend.mlx.core import _to_mlx_dtype
from keras.src.backend.mlx.core import convert_to_tensor


def rot90(array, k=1, axes=(0, 1)):
    if array.ndim < 2:
        raise ValueError(
            "Input array must have at least 2 dimensions. "
            f"Received: array.ndim={array.ndim}"
        )
    if len(axes) != 2 or axes[0] == axes[1]:
        raise ValueError(
            f"Invalid axes: {axes}. Axes must be a tuple "
            "of two different dimensions."
        )
    array = convert_to_tensor(array)
    k = k % 4
    if k == 0:
        return array
    ax0, ax1 = axes
    if k == 1:
        array = _flip(array, axis=ax1)
        return mx.swapaxes(array, ax0, ax1)
    if k == 2:
        array = _flip(array, axis=ax0)
        return _flip(array, axis=ax1)
    # k == 3
    array = _flip(array, axis=ax0)
    return mx.swapaxes(array, ax0, ax1)


def add(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.add(x1, x2)


def einsum(subscripts, *operands, **kwargs):
    operands = tree.map_structure(convert_to_tensor, operands)
    dtypes_to_resolve = list(set(standardize_dtype(x.dtype) for x in operands))
    if len(dtypes_to_resolve) == 1 and dtypes_to_resolve[0] == "int8":
        result_dtype = "int32"
    else:
        result_dtype = dtypes.result_type(*dtypes_to_resolve)
    compute_dtype = result_dtype
    # mx.einsum is backed by matmul and only supports float types; for
    # bfloat16 and integer/bool result dtypes compute in float32.
    if (
        compute_dtype == "bfloat16"
        or "int" in compute_dtype
        or (compute_dtype == "bool")
    ):
        compute_dtype = "float32"
    operands = tree.map_structure(
        lambda x: x.astype(_to_mlx_dtype(compute_dtype)), operands
    )
    return mx.einsum(subscripts, *operands).astype(_to_mlx_dtype(result_dtype))


def subtract(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.subtract(x1, x2)


def matmul(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    x1_dtype = standardize_dtype(x1.dtype)
    x2_dtype = standardize_dtype(x2.dtype)
    if x1_dtype == "int8" and x2_dtype == "int8":
        dtype = "int32"
    else:
        dtype = dtypes.result_type(x1.dtype, x2.dtype)
    # mx.matmul only supports float types; for integer/bool result dtypes
    # compute in float and cast the result back.
    compute_dtype = dtypes.result_type(dtype, float)
    x1 = x1.astype(_to_mlx_dtype(compute_dtype))
    x2 = x2.astype(_to_mlx_dtype(compute_dtype))
    return mx.matmul(x1, x2).astype(_to_mlx_dtype(dtype))


def multiply(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    # Accumulate in float32 to avoid low-precision (e.g. float16) overflow,
    # then cast back to the original dtype, matching numpy/jax/torch.
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    if "int" in ori_dtype or ori_dtype == "bool":
        result_dtype = compute_dtype
    else:
        result_dtype = ori_dtype
    output = mx.mean(
        x.astype(_to_mlx_dtype(compute_dtype)), axis=axis, keepdims=keepdims
    )
    return output.astype(_to_mlx_dtype(result_dtype))


def max(x, axis=None, keepdims=False, initial=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    if x.size == 0 and initial is not None:
        return convert_to_tensor(initial, dtype=standardize_dtype(x.dtype))
    result = mx.max(x, axis=axis, keepdims=keepdims)
    if initial is not None:
        initial = convert_to_tensor(initial, dtype=standardize_dtype(x.dtype))
        result = mx.maximum(result, initial)
    return result


def ones(shape, dtype=None):
    dtype = dtype or config.floatx()
    return mx.ones(shape, dtype=_to_mlx_dtype(dtype))


def zeros(shape, dtype=None):
    dtype = dtype or config.floatx()
    return mx.zeros(shape, dtype=_to_mlx_dtype(dtype))


def absolute(x):
    x = convert_to_tensor(x)
    return mx.abs(x)


def abs(x):
    return absolute(x)


def fabs(x):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.abs(x)


def all(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    return mx.all(x, axis=axis, keepdims=keepdims)


def allclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    if equal_nan:
        both_nan = mx.logical_and(mx.isnan(x1), mx.isnan(x2))
        close = mx.abs(x1 - x2) <= (atol + rtol * mx.abs(x2))
        return mx.all(mx.logical_or(both_nan, close))
    return mx.allclose(x1, x2, rtol=rtol, atol=atol)


def angle(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    # For real numbers, angle is 0 for positive, pi for negative
    if standardize_dtype(x.dtype) in ("complex64",):
        return mx.arctan2(mx.imag(x), mx.real(x))
    return mx.where(
        x < 0,
        convert_to_tensor(math.pi, dtype=dtype),
        convert_to_tensor(0.0, dtype=dtype),
    )


def any(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    return mx.any(x, axis=axis, keepdims=keepdims)


def amax(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    return mx.max(x, axis=axis, keepdims=keepdims)


def amin(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    return mx.min(x, axis=axis, keepdims=keepdims)


def append(x1, x2, axis=None):
    axis = standardize_axis_for_numpy(axis)
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    if axis is None:
        x1 = mx.flatten(x1)
        x2 = mx.flatten(x2)
        return mx.concatenate([x1, x2], axis=0)
    return mx.concatenate([x1, x2], axis=axis)


def arange(start, stop=None, step=None, dtype=None):
    if dtype is None:
        dtypes_to_resolve = [getattr(start, "dtype", type(start))]
        if stop is not None:
            dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
        if step is not None:
            dtypes_to_resolve.append(getattr(step, "dtype", type(step)))
        dtype = dtypes.result_type(*dtypes_to_resolve)
    if stop is None:
        start, stop = 0, start
    if step is None:
        step = 1
    return mx.arange(start, stop, step=step, dtype=_to_mlx_dtype(dtype))


def arccos(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.arccos(x)


def arccosh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.arccosh(x)


def arcsin(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.arcsin(x)


def arcsinh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.arcsinh(x)


def arctan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.arctan(x)


def arctan2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    return mx.arctan2(x1, x2)


def arctanh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.arctanh(x)


def argmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = standardize_axis_for_numpy(axis)
    if axis is None:
        x_flat = mx.flatten(x)
        result = mx.argmax(x_flat)
    else:
        result = mx.argmax(x, axis=axis)
    result = result.astype(mx.int32)
    if keepdims and axis is not None:
        result = mx.expand_dims(result, axis=axis)
    elif keepdims and axis is None:
        shape = [1] * x.ndim
        result = mx.reshape(result, shape)
    return result


def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    axis = standardize_axis_for_numpy(axis)
    if axis is None:
        x_flat = mx.flatten(x)
        result = mx.argmin(x_flat)
    else:
        result = mx.argmin(x, axis=axis)
    result = result.astype(mx.int32)
    if keepdims and axis is not None:
        result = mx.expand_dims(result, axis=axis)
    elif keepdims and axis is None:
        shape = [1] * x.ndim
        result = mx.reshape(result, shape)
    return result


def argsort(x, axis=-1):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    if x.ndim == 0:
        return mx.argsort(x, axis=None).astype(mx.int32)
    return mx.argsort(x, axis=axis).astype(mx.int32)


def array(x, dtype=None):
    return convert_to_tensor(x, dtype=dtype)


def view(x, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        return x
    mlx_dtype = _to_mlx_dtype(dtype)
    return mx.view(x, mlx_dtype)


def average(x, axis=None, weights=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtypes_to_resolve = [x.dtype, float]
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
    dtype = dtypes.result_type(*dtypes_to_resolve)
    x = x.astype(_to_mlx_dtype(dtype))
    if weights is not None:
        weights = weights.astype(_to_mlx_dtype(dtype))
        if weights.ndim == 1 and x.ndim > 1 and axis is not None:
            # Broadcast weights along the specified axis
            shape = [1] * x.ndim
            shape[axis] = weights.shape[0]
            weights = mx.reshape(weights, shape)
        sum_weights = mx.sum(weights * mx.ones_like(x), axis=axis)
        return mx.sum(x * weights, axis=axis) / sum_weights
    return mx.mean(x, axis=axis)


def bartlett(M):
    M = int(convert_to_tensor(M).item()) if not isinstance(M, int) else M
    return mx.bartlett(M).astype(_to_mlx_dtype(config.floatx()))


def hamming(M):
    M = int(convert_to_tensor(M).item()) if not isinstance(M, int) else M
    return mx.hamming(M).astype(_to_mlx_dtype(config.floatx()))


def hanning(M):
    M = int(convert_to_tensor(M).item()) if not isinstance(M, int) else M
    return mx.hanning(M).astype(_to_mlx_dtype(config.floatx()))


def heaviside(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype in ["int64"]:
        dtype = "float64"
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    return mx.where(
        x1 < 0,
        convert_to_tensor(0, dtype=dtype),
        mx.where(
            x1 == 0,
            x2,
            convert_to_tensor(1, dtype=dtype),
        ),
    )


def kaiser(M, beta):
    M = int(convert_to_tensor(M).item()) if not isinstance(M, int) else M
    if M == 0:
        return convert_to_tensor([], dtype=config.floatx())
    if M == 1:
        return ones(1, dtype=config.floatx())
    # Kaiser window:
    # I0(beta * sqrt(1 - ((n - (M-1)/2) / ((M-1)/2))^2)) / I0(beta)
    n = mx.arange(M, dtype=_to_mlx_dtype(config.floatx()))
    alpha = (M - 1) / 2.0
    arg = beta * mx.sqrt(
        mx.maximum(
            convert_to_tensor(0.0),
            1.0 - ((n - alpha) / alpha) ** 2,
        )
    )
    return i0(arg) / i0(convert_to_tensor(beta))


def bincount(x, weights=None, minlength=0, sparse=False):
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with mlx backend")
    x = convert_to_tensor(x)
    dtypes_to_resolve = [x.dtype]
    if weights is not None:
        weights = convert_to_tensor(weights)
        dtypes_to_resolve.append(weights.dtype)
        dtype = dtypes.result_type(*dtypes_to_resolve)
    else:
        dtype = "int32"

    def _bincount_1d(x_1d, w_1d, minlength):
        mx.eval(x_1d)
        max_val = int(mx.max(x_1d)) + 1 if x_1d.size > 0 else 0
        length = builtins.max(max_val, minlength)
        if length == 0:
            return mx.array([], dtype=_to_mlx_dtype(dtype))
        # One-hot via broadcast: x_1d[:, None] == arange[None, :]
        bins = mx.arange(length)
        one_hot = mx.expand_dims(x_1d, 1) == mx.expand_dims(bins, 0)
        if w_1d is not None:
            result = mx.sum(
                one_hot.astype(w_1d.dtype) * mx.expand_dims(w_1d, 1),
                axis=0,
            )
        else:
            result = mx.sum(one_hot.astype(mx.int32), axis=0)
        return result.astype(_to_mlx_dtype(dtype))

    if len(x.shape) == 2:
        rows = [
            _bincount_1d(
                x[i],
                weights[i] if weights is not None else None,
                minlength,
            )
            for i in range(x.shape[0])
        ]
        # Pad to same length
        mx.eval(*rows)
        max_len = builtins.max(r.shape[0] for r in rows)
        padded = [
            mx.pad(r, [(0, max_len - r.shape[0])])
            if r.shape[0] < max_len
            else r
            for r in rows
        ]
        return mx.stack(padded)
    return _bincount_1d(x, weights, minlength)


def bitwise_and(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = x.astype(_to_mlx_dtype(dtype))
    y = y.astype(_to_mlx_dtype(dtype))
    return mx.bitwise_and(x, y)


def bitwise_invert(x):
    x = convert_to_tensor(x)
    return mx.bitwise_invert(x)


def bitwise_not(x):
    return bitwise_invert(x)


def bitwise_or(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = x.astype(_to_mlx_dtype(dtype))
    y = y.astype(_to_mlx_dtype(dtype))
    return mx.bitwise_or(x, y)


def bitwise_xor(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)
    dtype = dtypes.result_type(x.dtype, y.dtype)
    x = x.astype(_to_mlx_dtype(dtype))
    y = y.astype(_to_mlx_dtype(dtype))
    return mx.bitwise_xor(x, y)


def bitwise_left_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)
        dtype = dtypes.result_type(x.dtype, y.dtype)
        x = x.astype(_to_mlx_dtype(dtype))
        y = y.astype(_to_mlx_dtype(dtype))
    return mx.left_shift(x, y)


def left_shift(x, y):
    return bitwise_left_shift(x, y)


def bitwise_right_shift(x, y):
    x = convert_to_tensor(x)
    if not isinstance(y, int):
        y = convert_to_tensor(y)
        dtype = dtypes.result_type(x.dtype, y.dtype)
        x = x.astype(_to_mlx_dtype(dtype))
        y = y.astype(_to_mlx_dtype(dtype))
    return mx.right_shift(x, y)


def right_shift(x, y):
    return bitwise_right_shift(x, y)


def blackman(M):
    M = int(convert_to_tensor(M).item()) if not isinstance(M, int) else M
    return mx.blackman(M).astype(_to_mlx_dtype(config.floatx()))


def broadcast_to(x, shape):
    x = convert_to_tensor(x)
    return mx.broadcast_to(x, shape)


def cbrt(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ["bool", "int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype == "int64":
        dtype = "float64"
    x = x.astype(_to_mlx_dtype(dtype))
    # cbrt(x) = sign(x) * |x|^(1/3)
    return mx.sign(x) * mx.power(
        mx.abs(x), convert_to_tensor(1.0 / 3.0, dtype=dtype)
    )


def ceil(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.ceil(x)


def clip(x, x_min, x_max):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype == "bool":
        dtype = "int32"
        x = x.astype(_to_mlx_dtype(dtype))
    return mx.clip(x, x_min, x_max)


def concatenate(xs, axis=0):
    axis = standardize_axis_for_numpy(axis)
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = tree.map_structure(
            lambda x: convert_to_tensor(x).astype(_to_mlx_dtype(dtype)), xs
        )
    else:
        xs = [convert_to_tensor(x) for x in xs]
    return mx.concatenate(xs, axis=axis)


def conjugate(x):
    x = convert_to_tensor(x)
    return mx.conjugate(x)


def conj(x):
    return conjugate(x)


def copy(x):
    x = convert_to_tensor(x)
    # MLX uses lazy evaluation; adding 0 creates a new node
    return mx.array(x)


def cos(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.cos(x)


def cosh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.cosh(x)


def count_nonzero(x, axis=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    mask = (x != 0).astype(mx.int32)
    return mx.sum(mask, axis=axis).astype(mx.int32)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    # Move the specified axes to the last position for mx.linalg.cross
    if axis is not None:
        axisa = axis
        axisb = axis
        axisc = axis
    x1 = mx.moveaxis(x1, axisa, -1)
    x2 = mx.moveaxis(x2, axisb, -1)
    if x1.shape[-1] == 2 and x2.shape[-1] == 2:
        # numpy returns only the z-component (drops the last dim) when both
        # input vectors have length 2.
        return x1[..., 0] * x2[..., 1] - x1[..., 1] * x2[..., 0]
    result = mx.linalg.cross(x1, x2, axis=-1)
    result = mx.moveaxis(result, -1, axisc)
    return result


def cumprod(x, axis=None, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    x = x.astype(_to_mlx_dtype(dtype))
    if axis is None:
        x = mx.flatten(x)
        axis = 0
    return mx.cumprod(x, axis=axis)


def cumsum(x, axis=None, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    x = x.astype(_to_mlx_dtype(dtype))
    if axis is None:
        x = mx.flatten(x)
        axis = 0
    return mx.cumsum(x, axis=axis)


def deg2rad(x):
    x = convert_to_tensor(x)
    dtype_str = standardize_dtype(x.dtype)
    if dtype_str in ["int64", "float64"]:
        dtype = "float64"
    elif dtype_str in ["bfloat16", "float16"]:
        dtype = dtype_str
    else:
        dtype = config.floatx()
    x = x.astype(_to_mlx_dtype(dtype))
    return x * convert_to_tensor(math.pi / 180.0, dtype=dtype)


def rad2deg(x):
    x = convert_to_tensor(x)
    dtype_str = standardize_dtype(x.dtype)
    if dtype_str in ["int64", "float64"]:
        dtype = "float64"
    elif dtype_str in ["bfloat16", "float16"]:
        dtype = dtype_str
    else:
        dtype = config.floatx()
    x = x.astype(_to_mlx_dtype(dtype))
    return x * convert_to_tensor(180.0 / math.pi, dtype=dtype)


def diag(x, k=0):
    x = convert_to_tensor(x)
    if x.ndim == 1:
        # Build a 2D matrix with x on the k-th diagonal
        n = x.shape[0] + builtins.abs(k)
        result = mx.zeros((n, n), dtype=x.dtype)
        if k >= 0:
            for i in range(x.shape[0]):
                result = result.at[i, i + k].add(x[i])
        else:
            for i in range(x.shape[0]):
                result = result.at[i - k, i].add(x[i])
        return result
    else:
        # Extract the k-th diagonal
        return mx.diagonal(x, offset=k)


def diagflat(x, k=0):
    x = convert_to_tensor(x)
    x = mx.flatten(x)
    return diag(x, k=k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    axis1 = standardize_axis_for_numpy(axis1)
    axis2 = standardize_axis_for_numpy(axis2)
    x = convert_to_tensor(x)
    return mx.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


def diff(a, n=1, axis=-1):
    a = convert_to_tensor(a)
    for _ in range(n):
        nd = a.ndim
        ax = axis
        if ax < 0:
            ax += nd
        slices_a = [builtins.slice(None)] * nd
        slices_b = [builtins.slice(None)] * nd
        slices_a[ax] = builtins.slice(1, None)
        slices_b[ax] = builtins.slice(None, -1)
        a = a[tuple(slices_a)] - a[tuple(slices_b)]
    return a


def digitize(x, bins):
    x = convert_to_tensor(x)
    bins = convert_to_tensor(bins)
    # digitize is equivalent to searchsorted with side='right' for
    # monotonically increasing bins, side='left' for decreasing
    mx.eval(bins)
    if bins.shape[0] > 1 and float(bins[0]) > float(bins[-1]):
        # Decreasing bins: reverse and adjust
        bins_rev = bins[::-1]
        result = bins.shape[0] - searchsorted(bins_rev, x, side="right")
    else:
        result = searchsorted(bins, x, side="right")
    return result.astype(_to_mlx_dtype("int32"))


def dot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if x1.ndim == 0 or x2.ndim == 0:
        x1 = x1.astype(_to_mlx_dtype(dtype))
        x2 = x2.astype(_to_mlx_dtype(dtype))
        return x1 * x2
    # mx.inner/mx.matmul/mx.tensordot only support float types; for
    # integer/bool result dtypes compute in float and cast back.
    compute_dtype = dtypes.result_type(dtype, float)
    x1 = x1.astype(_to_mlx_dtype(compute_dtype))
    x2 = x2.astype(_to_mlx_dtype(compute_dtype))
    if x1.ndim == 1 and x2.ndim == 1:
        result = mx.inner(x1, x2)
    elif x2.ndim == 1:
        result = mx.tensordot(x1, x2, axes=1)
    elif x1.ndim == 2 and x2.ndim == 2:
        result = mx.matmul(x1, x2)
    else:
        # General nD case: sum over last axis of x1 and second-to-last of x2
        result = mx.tensordot(x1, x2, axes=([-1], [-2]))
    return result.astype(_to_mlx_dtype(dtype))


def dstack(xs):
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = [convert_to_tensor(x).astype(_to_mlx_dtype(dtype)) for x in xs]
    else:
        xs = [convert_to_tensor(x) for x in xs]
    # Ensure at least 3D
    processed = []
    for x in xs:
        if x.ndim == 0:
            x = mx.reshape(x, (1, 1, 1))
        elif x.ndim == 1:
            x = mx.reshape(x, (1, x.shape[0], 1))
        elif x.ndim == 2:
            x = mx.expand_dims(x, axis=2)
        processed.append(x)
    return mx.concatenate(processed, axis=2)


def empty(shape, dtype=None):
    dtype = dtype or config.floatx()
    # MLX doesn't have empty, use zeros instead
    return mx.zeros(shape, dtype=_to_mlx_dtype(dtype))


def empty_like(x, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        dtype = standardize_dtype(x.dtype)
    return mx.zeros(x.shape, dtype=_to_mlx_dtype(dtype))


def equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.equal(x1, x2)


def exp(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = x.astype(_to_mlx_dtype(config.floatx()))
    return mx.exp(x)


def exp2(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = x.astype(_to_mlx_dtype(config.floatx()))
    # 2^x = exp(x * ln(2))
    return mx.exp(
        x * convert_to_tensor(math.log(2), dtype=standardize_dtype(x.dtype))
    )


def expand_dims(x, axis):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    return mx.expand_dims(x, axis=axis)


def expm1(x):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = x.astype(_to_mlx_dtype(config.floatx()))
    return mx.expm1(x)


def flip(x, axis=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    if axis is None:
        return _flip(x)
    if isinstance(axis, int):
        axis = (axis,)
    return _flip(x, axis=tuple(axis))


def fliplr(x):
    x = convert_to_tensor(x)
    return _flip(x, axis=(1,))


def flipud(x):
    x = convert_to_tensor(x)
    return _flip(x, axis=(0,))


def floor(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.floor(x)


def full(shape, fill_value, dtype=None):
    dtype = dtype or config.floatx()
    fill_value = convert_to_tensor(fill_value, dtype=dtype)
    return mx.full(shape, fill_value, dtype=_to_mlx_dtype(dtype))


def full_like(x, fill_value, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        dtype = standardize_dtype(x.dtype)
    # Pass a typed fill value (like full() does) so mlx honors the dtype;
    # a raw Python int would make mlx infer int32 and drop a bool dtype.
    fill_value = convert_to_tensor(fill_value, dtype=dtype)
    return mx.full(x.shape, fill_value, dtype=_to_mlx_dtype(dtype))


def gcd(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = mx.abs(x1.astype(_to_mlx_dtype(dtype)))
    x2 = mx.abs(x2.astype(_to_mlx_dtype(dtype)))
    # Unrolled Euclidean algorithm — converges in log2(max_value) steps
    n_iters = _to_mlx_dtype(dtype).size * 8
    for _ in range(n_iters):
        safe_x2 = mx.where(x2 == 0, mx.array(1, dtype=x2.dtype), x2)
        r = mx.remainder(x1, safe_x2)
        done = x2 == 0
        x1 = mx.where(done, x1, x2)
        x2 = mx.where(done, mx.zeros_like(x2), r)
    return x1


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    if axis != 0:
        raise ValueError(
            "mx.geomspace does not support an `axis` argument. "
            f"Received axis={axis}"
        )
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    mlx_dtype = _to_mlx_dtype(dtype)

    start = convert_to_tensor(start, dtype=dtype)
    stop = convert_to_tensor(stop, dtype=dtype)

    log_start = mx.log10(mx.abs(start))
    log_stop = mx.log10(mx.abs(stop))

    result = logspace(
        log_start, log_stop, num=num, endpoint=endpoint, base=10, dtype=dtype
    )
    return (result * mx.sign(start)).astype(mlx_dtype)


def greater(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.greater(x1, x2)


def greater_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.greater_equal(x1, x2)


def hstack(xs):
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = [convert_to_tensor(x).astype(_to_mlx_dtype(dtype)) for x in xs]
    else:
        xs = [convert_to_tensor(x) for x in xs]
    # hstack: concatenate along axis=1 for 2D+, axis=0 for 1D
    if xs[0].ndim <= 1:
        # Ensure 1D
        xs = [mx.flatten(x) if x.ndim == 0 else x for x in xs]
        return mx.concatenate(xs, axis=0)
    return mx.concatenate(xs, axis=1)


def hsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    if x.ndim < 1:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    axis = 1 if x.ndim > 1 else 0
    return split(x, indices_or_sections, axis=axis)


def hypot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
        dtype = config.floatx()
    elif dtype in ["int64"]:
        dtype = "float64"
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    return mx.sqrt(x1 * x1 + x2 * x2)


def identity(n, dtype=None):
    dtype = dtype or config.floatx()
    return mx.eye(n, dtype=_to_mlx_dtype(dtype))


def imag(x):
    x = convert_to_tensor(x)
    return mx.imag(x)


def i0(x):
    x = convert_to_tensor(x)
    dtype = (
        "float64"
        if standardize_dtype(x.dtype) in ["int64", "float64"]
        else dtypes.result_type(x.dtype, float)
    )
    x = x.astype(_to_mlx_dtype(dtype))
    ax = mx.abs(x)
    # Abramowitz & Stegun polynomial approximation for I0
    # For |x| < 3.75
    t = (ax / 3.75) ** 2
    small = 1.0 + t * (
        3.5156229
        + t
        * (
            3.0899424
            + t
            * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))
        )
    )
    # For |x| >= 3.75
    t2 = 3.75 / ax
    large = (
        mx.exp(ax)
        / mx.sqrt(ax)
        * (
            0.39894228
            + t2
            * (
                0.01328592
                + t2
                * (
                    0.00225319
                    + t2
                    * (
                        -0.00157565
                        + t2
                        * (
                            0.00916281
                            + t2
                            * (
                                -0.02057706
                                + t2
                                * (
                                    0.02635537
                                    + t2 * (-0.01647633 + t2 * 0.00392377)
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return mx.where(ax < 3.75, small, large)


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    close = mx.abs(x1 - x2) <= (atol + rtol * mx.abs(x2))
    if equal_nan:
        both_nan = mx.logical_and(mx.isnan(x1), mx.isnan(x2))
        return mx.logical_or(both_nan, close)
    return close


def isfinite(x):
    x = convert_to_tensor(x)
    return mx.isfinite(x)


def isin(x1, x2, assume_unique=False, invert=False):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    x2_flat = mx.flatten(x2)
    # Broadcast compare: x1[..., None] == x2_flat[None, ...]
    x1_flat = mx.flatten(x1)
    matches = mx.equal(mx.expand_dims(x1_flat, 1), mx.expand_dims(x2_flat, 0))
    result = mx.any(matches, axis=1)
    result = mx.reshape(result, x1.shape)
    if invert:
        result = mx.logical_not(result)
    return result


def isinf(x):
    x = convert_to_tensor(x)
    return mx.isinf(x)


def isnan(x):
    x = convert_to_tensor(x)
    return mx.isnan(x)


def isneginf(x):
    x = convert_to_tensor(x)
    return mx.logical_and(mx.isinf(x), x < 0)


def isposinf(x):
    x = convert_to_tensor(x)
    return mx.logical_and(mx.isinf(x), x > 0)


def isreal(x):
    x = convert_to_tensor(x)
    dtype_str = standardize_dtype(x.dtype)
    if "complex" not in dtype_str:
        return mx.ones(x.shape, dtype=mx.bool_)
    return mx.equal(mx.imag(x), convert_to_tensor(0))


def kron(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    return mx.kron(x1, x2)


def lcm(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    g = gcd(x1, x2)
    return mx.where(g == 0, mx.zeros_like(g), mx.abs(x1 * x2) // g)


def ldexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    if standardize_dtype(x2.dtype) not in dtypes.INT_TYPES:
        raise TypeError(
            f"ldexp exponent must be an integer type. "
            f"Received: x2 dtype={x2.dtype}"
        )
    # ldexp(x1, x2) = x1 * 2^x2
    x1 = x1.astype(_to_mlx_dtype(dtype))
    return x1 * mx.power(
        convert_to_tensor(2.0, dtype=dtype),
        x2.astype(_to_mlx_dtype(dtype)),
    )


def less(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.less(x1, x2)


def less_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    if axis != 0:
        raise ValueError(
            "mx.linspace does not support an `axis` argument. "
            f"Received axis={axis}"
        )
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    mlx_dtype = _to_mlx_dtype(dtype)

    step = float("nan")
    if endpoint:
        if num > 1:
            step = (stop - start) / (num - 1)
    else:
        if num > 0:
            step = (stop - start) / num
        if num > 1:
            stop = stop - ((stop - start) / num)

    if hasattr(start, "__len__") and hasattr(stop, "__len__"):
        start = convert_to_tensor(start, dtype=dtype)
        stop = convert_to_tensor(stop, dtype=dtype)
        steps = mx.arange(num, dtype=mlx_dtype) / (num - 1)

        # reshape `steps` to allow for broadcasting
        for _ in range(start.ndim):
            steps = mx.expand_dims(steps, -1)

        # increments from `start` to `stop` in each dimension
        linspace = start[None] + steps * (stop - start)[None]
    else:
        linspace = mx.linspace(start, stop, num=num, dtype=mlx_dtype)

    if retstep:
        return (linspace, step)
    return linspace


def log(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.log(x)


def log10(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.log10(x)


def log1p(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.log1p(x)


def log2(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.log2(x)


def logaddexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    return mx.logaddexp(x1, x2)


def logaddexp2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    # logaddexp2(x1, x2) = log2(2^x1 + 2^x2)
    ln2 = convert_to_tensor(math.log(2), dtype=dtype)
    return mx.logaddexp(x1 * ln2, x2 * ln2) / ln2


def logical_and(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.logical_and(x1, x2)


def logical_not(x):
    x = convert_to_tensor(x)
    return mx.logical_not(x)


def logical_or(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    if axis != 0:
        raise ValueError(
            "mx.logspace does not support an `axis` argument. "
            f"Received axis={axis}"
        )
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
            getattr(stop, "dtype", type(stop)),
            float,
        ]
        dtype = dtypes.result_type(*dtypes_to_resolve)
    mlx_dtype = _to_mlx_dtype(dtype)

    if endpoint is False:
        stop = stop - ((stop - start) / num)
    if hasattr(start, "__len__") and hasattr(stop, "__len__"):
        start = convert_to_tensor(start, dtype=dtype)
        stop = convert_to_tensor(stop, dtype=dtype)
        steps = mx.arange(num, dtype=mlx_dtype) / (num - 1)

        # reshape `steps` to allow for broadcasting
        for _ in range(start.ndim):
            steps = mx.expand_dims(steps, -1)

        linspace = start[None] + steps * (stop - start)[None]
        logspace = base**linspace
    else:
        linspace = mx.linspace(start, stop, num=num, dtype=mlx_dtype)
        logspace = mx.power(
            convert_to_tensor(base, dtype=dtype), linspace
        ).astype(mlx_dtype)
    return logspace


def maximum(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.maximum(x1, x2)


def fmax(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    return mx.where(
        mx.isnan(x1),
        x2,
        mx.where(mx.isnan(x2), x1, mx.maximum(x1, x2)),
    )


def median(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype, float)
    result = mx.median(x, axis=axis, keepdims=keepdims)
    return result.astype(_to_mlx_dtype(dtype))


def meshgrid(*x, indexing="xy"):
    x = [convert_to_tensor(xi) for xi in x]
    return mx.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    if x.size == 0 and initial is not None:
        return convert_to_tensor(initial, dtype=standardize_dtype(x.dtype))
    result = mx.min(x, axis=axis, keepdims=keepdims)
    if initial is not None:
        initial = convert_to_tensor(initial, dtype=standardize_dtype(x.dtype))
        result = mx.minimum(result, initial)
    return result


def minimum(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.minimum(x1, x2)


def fmin(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    return mx.where(
        mx.isnan(x1),
        x2,
        mx.where(mx.isnan(x2), x1, mx.minimum(x1, x2)),
    )


def mod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype == "bool":
        dtype = "int32"
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    return mx.remainder(x1, x2)


def fmod(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    if dtype == "bool":
        dtype = "int32"
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    quotient = x1 / x2
    truncated = mx.sign(quotient) * mx.floor(mx.abs(quotient))
    result = x1 - truncated * x2
    return result.astype(_to_mlx_dtype(dtype))


def moveaxis(x, source, destination):
    x = convert_to_tensor(x)
    if isinstance(source, int):
        source = (source,)
    if isinstance(destination, int):
        destination = (destination,)
    ndim = x.ndim
    source = [s % ndim for s in source]
    destination = [d % ndim for d in destination]
    # Build permutation
    order = [n for n in range(ndim) if n not in source]
    for dst, src in sorted(zip(destination, source)):
        order.insert(dst, src)
    return mx.transpose(x, order)


def nanargmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype_str = standardize_dtype(x.dtype)
    if "float" not in dtype_str:
        return argmax(x, axis=axis, keepdims=keepdims)
    nan_mask = mx.isnan(x)
    x_filled = mx.where(
        nan_mask, convert_to_tensor(float("-inf"), dtype=dtype_str), x
    )
    all_nan = mx.all(nan_mask, axis=axis, keepdims=keepdims)
    result = argmax(x_filled, axis=axis, keepdims=keepdims)
    return mx.where(all_nan, convert_to_tensor(-1, dtype="int32"), result)


def nanargmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype_str = standardize_dtype(x.dtype)
    if "float" not in dtype_str:
        return argmin(x, axis=axis, keepdims=keepdims)
    nan_mask = mx.isnan(x)
    x_filled = mx.where(
        nan_mask, convert_to_tensor(float("inf"), dtype=dtype_str), x
    )
    all_nan = mx.all(nan_mask, axis=axis, keepdims=keepdims)
    result = argmin(x_filled, axis=axis, keepdims=keepdims)
    return mx.where(all_nan, convert_to_tensor(-1, dtype="int32"), result)


def nancumsum(x, axis=None, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    x = mx.where(mx.isnan(x), convert_to_tensor(0, dtype=x.dtype), x)
    x = x.astype(_to_mlx_dtype(dtype))
    if axis is None:
        x = mx.flatten(x)
        axis = 0
    return mx.cumsum(x, axis=axis)


def nancumprod(x, axis=None, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    if dtype == "bool":
        dtype = "int32"
    x = mx.where(mx.isnan(x), convert_to_tensor(1, dtype=x.dtype), x)
    x = x.astype(_to_mlx_dtype(dtype))
    if axis is None:
        x = mx.flatten(x)
        axis = 0
    return mx.cumprod(x, axis=axis)


def nanmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    # Integer/bool inputs can never contain NaN; the inf sentinel below would
    # OverflowError when cast to an integer dtype, so short-circuit to mx.max.
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        return mx.max(x, axis=axis, keepdims=keepdims)
    all_nan = mx.all(mx.isnan(x), axis=axis, keepdims=keepdims)
    x_filled = mx.where(
        mx.isnan(x), convert_to_tensor(float("-inf"), dtype=x.dtype), x
    )
    result = mx.max(x_filled, axis=axis, keepdims=keepdims)
    return mx.where(
        all_nan, convert_to_tensor(float("nan"), dtype=result.dtype), result
    )


def nanmean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(standardize_dtype(x.dtype), float)
    x = x.astype(_to_mlx_dtype(dtype))
    nan_mask = mx.isnan(x)
    x_filled = mx.where(nan_mask, convert_to_tensor(0, dtype=dtype), x)
    count = mx.sum(
        mx.logical_not(nan_mask).astype(_to_mlx_dtype(dtype)),
        axis=axis,
        keepdims=keepdims,
    )
    return mx.sum(x_filled, axis=axis, keepdims=keepdims) / count


def nanmedian(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    if axis == () or axis == []:
        return x
    dtype = dtypes.result_type(standardize_dtype(x.dtype), float)
    x = x.astype(_to_mlx_dtype(dtype))
    nan_mask = mx.isnan(x)
    reduced = _quantile_reduced_axes(axis, x.ndim)
    n_valid = mx.sum(
        (~nan_mask).astype(mx.int32), axis=reduced, keepdims=True
    ).astype(x.dtype)
    all_nan = n_valid == 0
    n_safe = mx.where(all_nan, convert_to_tensor(1, dtype=x.dtype), n_valid)
    # Replace NaN with +inf so they sort to the end; count of real values
    # comes from `nan_mask`, not from the sentinel (genuine +inf must remain
    # a valid value).
    x_filled = mx.where(
        nan_mask, convert_to_tensor(float("inf"), dtype=x.dtype), x
    )
    work, other_shape = _quantile_to_work(x_filled, reduced)
    n_safe = mx.reshape(n_safe, other_shape + [1])
    all_nan = mx.reshape(all_nan, other_shape + [1])
    sorted_x = mx.sort(work, axis=-1)
    indices = 0.5 * (n_safe - 1)

    def gather(arr, idx):
        return mx.take_along_axis(arr, idx, axis=-1)

    result = _quantile_interp(sorted_x, indices, "linear", gather)
    result = mx.where(
        all_nan, convert_to_tensor(float("nan"), dtype=result.dtype), result
    )
    result = mx.squeeze(result, axis=-1)
    if keepdims:
        target = [1 if a in reduced else x.shape[a] for a in range(x.ndim)]
        result = mx.reshape(result, target)
    return result


def nanmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    # Integer/bool inputs can never contain NaN; the inf sentinel below would
    # OverflowError when cast to an integer dtype, so short-circuit to mx.min.
    if x.dtype not in (mx.float16, mx.bfloat16, mx.float32):
        return mx.min(x, axis=axis, keepdims=keepdims)
    all_nan = mx.all(mx.isnan(x), axis=axis, keepdims=keepdims)
    x_filled = mx.where(
        mx.isnan(x), convert_to_tensor(float("inf"), dtype=x.dtype), x
    )
    result = mx.min(x_filled, axis=axis, keepdims=keepdims)
    return mx.where(
        all_nan, convert_to_tensor(float("nan"), dtype=result.dtype), result
    )


def nanpercentile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype == "bool":
        x = x.astype(_to_mlx_dtype(config.floatx()))
    dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    q = convert_to_tensor(q, dtype=dtype) / 100
    x_filled = mx.where(
        mx.isnan(x),
        convert_to_tensor(float("inf"), dtype=x.dtype),
        x,
    )
    return _nanquantile_impl(
        x_filled, q, axis=axis, method=method, keepdims=keepdims
    )


def nanprod(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(x.dtype)
    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    x = mx.where(mx.isnan(x), convert_to_tensor(1, dtype=x.dtype), x)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.prod(x, axis=axis, keepdims=keepdims)


def nanquantile(x, q, axis=None, method="linear", keepdims=False):
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype == "bool":
        x = x.astype(_to_mlx_dtype(config.floatx()))
    dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    q = convert_to_tensor(q, dtype=dtype)
    # Replace NaN with +inf so they sort to the end
    x_filled = mx.where(
        mx.isnan(x),
        convert_to_tensor(float("inf"), dtype=x.dtype),
        x,
    )
    return _nanquantile_impl(
        x_filled, q, axis=axis, method=method, keepdims=keepdims
    )


def nanstd(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(compute_dtype))
    nan_mask = mx.isnan(x)
    x_filled = mx.where(nan_mask, convert_to_tensor(0, dtype=compute_dtype), x)
    count = mx.sum(
        mx.logical_not(nan_mask).astype(_to_mlx_dtype(compute_dtype)),
        axis=axis,
        keepdims=keepdims,
    )
    mean_val = mx.sum(x_filled, axis=axis, keepdims=keepdims) / count
    if not keepdims and axis is not None:
        mean_expanded = mx.expand_dims(mean_val, axis=axis)
    elif not keepdims and axis is None:
        mean_expanded = mean_val
    else:
        mean_expanded = mean_val
    diff_sq = mx.where(
        nan_mask,
        convert_to_tensor(0, dtype=compute_dtype),
        (x - mean_expanded) ** 2,
    )
    variance = mx.sum(diff_sq, axis=axis, keepdims=keepdims) / count
    return mx.sqrt(variance).astype(_to_mlx_dtype(result_dtype))


def nansum(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    x = mx.where(mx.isnan(x), convert_to_tensor(0, dtype=x.dtype), x)
    return mx.sum(x, axis=axis, keepdims=keepdims).astype(_to_mlx_dtype(dtype))


def nanvar(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(compute_dtype))
    nan_mask = mx.isnan(x)
    x_filled = mx.where(nan_mask, convert_to_tensor(0, dtype=compute_dtype), x)
    count = mx.sum(
        mx.logical_not(nan_mask).astype(_to_mlx_dtype(compute_dtype)),
        axis=axis,
        keepdims=keepdims,
    )
    mean_val = mx.sum(x_filled, axis=axis, keepdims=keepdims) / count
    if not keepdims and axis is not None:
        mean_expanded = mx.expand_dims(mean_val, axis=axis)
    elif not keepdims and axis is None:
        mean_expanded = mean_val
    else:
        mean_expanded = mean_val
    diff_sq = mx.where(
        nan_mask,
        convert_to_tensor(0, dtype=compute_dtype),
        (x - mean_expanded) ** 2,
    )
    return (mx.sum(diff_sq, axis=axis, keepdims=keepdims) / count).astype(
        _to_mlx_dtype(result_dtype)
    )


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    x = convert_to_tensor(x)
    return mx.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def ndim(x):
    x = convert_to_tensor(x)
    return x.ndim


def nonzero(x):
    x = convert_to_tensor(x)
    # numpy has no bfloat16 dtype, so np.array() on a bfloat16 mlx array
    # raises; cast to float32 for the index computation (indices are integers).
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    # Need mx.eval for data-dependent shapes
    mx.eval(x)
    result = np.nonzero(np.array(x))
    return tuple(convert_to_tensor(idx, dtype="int32") for idx in result)


def not_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.not_equal(x1, x2)


def zeros_like(x, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        dtype = standardize_dtype(x.dtype)
    return mx.zeros_like(x).astype(_to_mlx_dtype(dtype))


def ones_like(x, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        dtype = standardize_dtype(x.dtype)
    return mx.ones_like(x).astype(_to_mlx_dtype(dtype))


def outer(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    return mx.outer(x1, x2)


def pad(x, pad_width, mode="constant", constant_values=None):
    x = convert_to_tensor(x)
    if mode != "constant" and constant_values is not None:
        raise ValueError(
            "Argument `constant_values` can only be "
            "provided when `mode == 'constant'`. "
            f"Received: mode={mode}"
        )
    if mode == "constant":
        if constant_values is None:
            constant_values = 0
        # Normalize pad_width to list of (before, after) tuples
        if isinstance(pad_width, int):
            pad_width = [(pad_width, pad_width)] * x.ndim
        elif isinstance(pad_width, (list, tuple)):
            if isinstance(pad_width[0], int):
                pad_width = [tuple(pad_width)] * x.ndim
        return mx.pad(x, pad_width, constant_values=constant_values)
    # For other modes, implement via slicing + concatenate
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width)] * x.ndim
    elif isinstance(pad_width, (list, tuple)):
        if isinstance(pad_width[0], int):
            pad_width = [tuple(pad_width)] * x.ndim
    for dim in range(x.ndim):
        before, after = pad_width[dim]
        if before == 0 and after == 0:
            continue
        n = x.shape[dim]
        if mode == "reflect":
            # Reflect without the edge: [before:0:-1]
            if before > 0:
                slc = [slice(None)] * x.ndim
                slc[dim] = slice(before, 0, -1)
                x = mx.concatenate([x[tuple(slc)], x], axis=dim)
            if after > 0:
                slc = [slice(None)] * x.ndim
                slc[dim] = slice(-2, -2 - after, -1)
                x = mx.concatenate([x, x[tuple(slc)]], axis=dim)
        elif mode == "symmetric":
            # Reflect including the edge: [before-1::-1]
            if before > 0:
                slc = [slice(None)] * x.ndim
                slc[dim] = (
                    slice(before - 1, None, -1)
                    if before <= n
                    else slice(n - 1, None, -1)
                )
                pad_arr = x[tuple(slc)]
                # Trim if needed
                trim = [slice(None)] * x.ndim
                trim[dim] = slice(0, before)
                x = mx.concatenate([pad_arr[tuple(trim)], x], axis=dim)
            if after > 0:
                slc = [slice(None)] * x.ndim
                cur_n = x.shape[dim]
                slc[dim] = slice(
                    cur_n - 1,
                    cur_n - 1 - after if cur_n - 1 - after >= 0 else None,
                    -1,
                )
                pad_arr = x[tuple(slc)]
                trim = [slice(None)] * x.ndim
                trim[dim] = slice(0, after)
                x = mx.concatenate([x, pad_arr[tuple(trim)]], axis=dim)
        elif mode == "wrap":
            if before > 0:
                slc = [slice(None)] * x.ndim
                slc[dim] = slice(-before, None)
                x = mx.concatenate([x[tuple(slc)], x], axis=dim)
            if after > 0:
                slc = [slice(None)] * x.ndim
                slc[dim] = slice(0, after)
                x = mx.concatenate([x, x[tuple(slc)]], axis=dim)
        else:
            raise ValueError(f"Unsupported pad mode: {mode}")
    return x


def percentile(x, q, axis=None, method="linear", keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype == "bool":
        x = x.astype(_to_mlx_dtype(config.floatx()))
    if ori_dtype == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    q = convert_to_tensor(q, dtype=dtype) / 100
    return _quantile_impl(x, q, axis=axis, method=method, keepdims=keepdims)


def prod(x, axis=None, keepdims=False, dtype=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    if dtype is None:
        dtype = dtypes.result_type(x.dtype)
        if dtype in ("bool", "int8", "int16"):
            dtype = "int32"
        elif dtype in ("uint8", "uint16"):
            dtype = "uint32"
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.prod(x, axis=axis, keepdims=keepdims)


def ptp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return mx.max(x, axis=axis, keepdims=keepdims) - mx.min(
        x, axis=axis, keepdims=keepdims
    )


def _quantile_reduced_axes(axis, ndim):
    if axis is None:
        return list(range(ndim))
    if isinstance(axis, (list, tuple)):
        return sorted(a % ndim for a in axis)
    return [axis % ndim]


def _quantile_to_work(x, reduced):
    ndim = x.ndim
    other = [a for a in range(ndim) if a not in reduced]
    perm = other + reduced
    xt = mx.transpose(x, perm)
    other_shape = [x.shape[a] for a in other]
    merged = 1
    for a in reduced:
        merged *= x.shape[a]
    return mx.reshape(xt, other_shape + [merged]), other_shape


def _quantile_interp(sorted_x, indices, method, gather):
    n = sorted_x.shape[-1]
    lo = mx.clip(mx.floor(indices).astype(mx.int32), 0, n - 1)
    hi = mx.clip(mx.ceil(indices).astype(mx.int32), 0, n - 1)
    if method == "nearest":
        idx = mx.clip(mx.round(indices).astype(mx.int32), 0, n - 1)
        return gather(sorted_x, idx)
    lo_vals = gather(sorted_x, lo)
    hi_vals = gather(sorted_x, hi)
    if method == "lower":
        return lo_vals
    if method == "higher":
        return hi_vals
    if method == "midpoint":
        return (lo_vals + hi_vals) / 2
    frac = indices - mx.floor(indices)
    return lo_vals + frac * (hi_vals - lo_vals)


def _quantile_finalize(result, q, k, orig_shape, reduced, scalar_q, keepdims):
    result = mx.moveaxis(result, result.ndim - 1, 0)
    if keepdims:
        target = [k] + [
            1 if a in reduced else orig_shape[a] for a in range(len(orig_shape))
        ]
        result = mx.reshape(result, target)
    if scalar_q:
        result = mx.squeeze(result, axis=0)
    return result


def _nanquantile_impl(x, q, axis=None, method="linear", keepdims=False):
    """Like _quantile_impl but uses non-inf count (inf = replaced NaN)."""
    scalar_q = q.ndim == 0
    if scalar_q:
        q = mx.expand_dims(q, 0)
    k = q.shape[0]
    orig_shape = list(x.shape)
    reduced = _quantile_reduced_axes(axis, x.ndim)
    work, other_shape = _quantile_to_work(x, reduced)
    sorted_x = mx.sort(work, axis=-1)

    n_valid = mx.sum(
        (sorted_x < float("inf")).astype(mx.int32), axis=-1, keepdims=True
    ).astype(sorted_x.dtype)
    all_nan = n_valid == 0
    n_safe = mx.where(
        all_nan, convert_to_tensor(1, dtype=sorted_x.dtype), n_valid
    )
    q_row = mx.reshape(q, [1] * len(other_shape) + [k])
    indices = q_row * (n_safe - 1)

    def gather(arr, idx):
        return mx.take_along_axis(arr, idx, axis=-1)

    result = _quantile_interp(sorted_x, indices, method, gather)
    result = mx.where(
        all_nan, convert_to_tensor(float("nan"), dtype=result.dtype), result
    )
    return _quantile_finalize(
        result, q, k, orig_shape, reduced, scalar_q, keepdims
    )


def _quantile_impl(x, q, axis=None, method="linear", keepdims=False):
    """Sort-based quantile with linear/lower/higher/midpoint/nearest, MLX."""
    scalar_q = q.ndim == 0
    if scalar_q:
        q = mx.expand_dims(q, 0)
    k = q.shape[0]
    orig_shape = list(x.shape)
    reduced = _quantile_reduced_axes(axis, x.ndim)
    work, _ = _quantile_to_work(x, reduced)
    sorted_x = mx.sort(work, axis=-1)

    indices = q * (sorted_x.shape[-1] - 1)

    def gather(arr, idx):
        return mx.take(arr, idx, axis=-1)

    result = _quantile_interp(sorted_x, indices, method, gather)
    return _quantile_finalize(
        result, q, k, orig_shape, reduced, scalar_q, keepdims
    )


def quantile(x, q, axis=None, method="linear", keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if ori_dtype == "bool":
        x = x.astype(_to_mlx_dtype(config.floatx()))
    if ori_dtype == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    q = convert_to_tensor(q, dtype=dtype)
    return _quantile_impl(x, q, axis=axis, method=method, keepdims=keepdims)


def ravel(x):
    x = convert_to_tensor(x)
    return mx.flatten(x)


def unravel_index(indices, shape):
    indices = convert_to_tensor(indices)
    dtype = dtypes.result_type(indices.dtype)
    result = []
    remainder = indices
    for dim in reversed(shape):
        result.append(
            mx.remainder(remainder, convert_to_tensor(dim, dtype=dtype))
        )
        remainder = mx.floor_divide(
            remainder, convert_to_tensor(dim, dtype=dtype)
        )
    result.reverse()
    return tuple(r.astype(_to_mlx_dtype(dtype)) for r in result)


def real(x):
    x = convert_to_tensor(x)
    return mx.real(x)


def reciprocal(x):
    x = convert_to_tensor(x)
    return mx.reciprocal(x)


def repeat(x, repeats, axis=None):
    x = convert_to_tensor(x)
    if isinstance(repeats, int):
        return mx.repeat(x, repeats, axis=axis)
    if axis is None:
        x = mx.flatten(x)
        axis = 0
    axis = axis % x.ndim
    n = x.shape[axis]
    repeats = convert_to_tensor(repeats, dtype="int32")
    if repeats.ndim == 0:
        return mx.repeat(x, int(repeats), axis=axis)
    if repeats.shape[0] == 1:
        repeats = mx.broadcast_to(repeats, (n,))
    # Build gather indices: position j maps to the source index equal to the
    # number of cumulative repeats not yet exceeding j.
    total = int(mx.sum(repeats))
    cumulative = mx.cumsum(repeats)
    positions = mx.arange(total)
    indices = mx.sum(
        (positions[:, None] >= cumulative[None, :]).astype(mx.int32), axis=1
    )
    return mx.take(x, indices, axis=axis)


def reshape(x, newshape):
    x = convert_to_tensor(x)
    if isinstance(newshape, int):
        newshape = (newshape,)
    return mx.reshape(x, newshape)


def roll(x, shift, axis=None):
    x = convert_to_tensor(x)
    return mx.roll(x, shift, axis=axis)


def searchsorted(sorted_sequence, values, side="left"):
    sorted_sequence = convert_to_tensor(sorted_sequence)
    values = convert_to_tensor(values)
    if ndim(sorted_sequence) != 1:
        raise ValueError(
            "`searchsorted` only supports 1-D sorted sequences. "
            "You can use `keras.ops.vectorized_map` "
            "to extend it to N-D sequences. Received: "
            f"sorted_sequence.shape={sorted_sequence.shape}"
        )
    out_type = "int32" if sorted_sequence.shape[0] <= 2**31 - 1 else "int64"
    # Broadcast comparison: expand values and compare against sorted_sequence
    vals_shape = values.shape
    vals_flat = mx.flatten(values)
    # vals_flat[:, None] vs sorted_sequence[None, :]
    if side == "left":
        cmp = mx.expand_dims(vals_flat, 1) > mx.expand_dims(sorted_sequence, 0)
    else:
        cmp = mx.expand_dims(vals_flat, 1) >= mx.expand_dims(sorted_sequence, 0)
    # Count True per row = insertion index
    result = mx.sum(cmp.astype(mx.int32), axis=1)
    result = mx.reshape(result, vals_shape) if vals_shape else result
    return result.astype(_to_mlx_dtype(out_type))


def sign(x):
    x = convert_to_tensor(x)
    return mx.sign(x)


def signbit(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "float" in dtype:
        # View as unsigned int to check sign bit — handles -0.0 correctly
        uint_dtype = {2: mx.uint16, 4: mx.uint32}[x.dtype.size]
        bits = x.view(uint_dtype)
        n_bits = x.dtype.size * 8
        return mx.right_shift(bits, n_bits - 1).astype(mx.bool_)
    return x < 0


def sin(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.sin(x)


def sinc(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    # sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1
    pi_x = x * convert_to_tensor(math.pi, dtype=dtype)
    return mx.where(
        x == 0,
        convert_to_tensor(1.0, dtype=dtype),
        mx.sin(pi_x) / pi_x,
    )


def sinh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.sinh(x)


def size(x):
    x = convert_to_tensor(x)
    result = 1
    for d in x.shape:
        result *= d
    return result


def sort(x, axis=-1):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    return mx.sort(x, axis=axis)


def split(x, indices_or_sections, axis=0):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    if isinstance(indices_or_sections, int):
        # Split into N equal sections
        size = x.shape[axis]
        if size % indices_or_sections != 0:
            raise ValueError(
                f"array split does not result in an equal division: "
                f"{size} / {indices_or_sections}"
            )
        section_size = size // indices_or_sections
        indices = [section_size * i for i in range(1, indices_or_sections)]
    else:
        indices = list(indices_or_sections)
    # Build slices from indices
    result = []
    prev = 0
    for idx in indices:
        idx = int(idx)
        slices = [builtins.slice(None)] * x.ndim
        slices[axis] = builtins.slice(prev, idx)
        result.append(x[tuple(slices)])
        prev = idx
    slices = [builtins.slice(None)] * x.ndim
    slices[axis] = builtins.slice(prev, None)
    result.append(x[tuple(slices)])
    return result


def array_split(x, indices_or_sections, axis=0):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    size = x.shape[axis]
    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        base_size = size // n
        remainder = size % n
        indices = []
        offset = 0
        for i in range(n):
            offset += base_size + (1 if i < remainder else 0)
            if i < n - 1:
                indices.append(offset)
    else:
        indices = list(indices_or_sections)
    result = []
    prev = 0
    for idx in indices:
        slices = [builtins.slice(None)] * x.ndim
        slices[axis] = builtins.slice(prev, idx)
        result.append(x[tuple(slices)])
        prev = idx
    slices = [builtins.slice(None)] * x.ndim
    slices[axis] = builtins.slice(prev, None)
    result.append(x[tuple(slices)])
    return result


def stack(x, axis=0):
    axis = standardize_axis_for_numpy(axis)
    dtype_set = set([getattr(a, "dtype", type(a)) for a in x])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        x = [convert_to_tensor(a).astype(_to_mlx_dtype(dtype)) for a in x]
    else:
        x = [convert_to_tensor(a) for a in x]
    return mx.stack(x, axis=axis)


def std(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    ori_dtype = standardize_dtype(x.dtype)
    if "int" in ori_dtype or ori_dtype == "bool":
        x = x.astype(_to_mlx_dtype(config.floatx()))
    return mx.std(x, axis=axis, keepdims=keepdims)


def swapaxes(x, axis1, axis2):
    x = convert_to_tensor(x)
    return mx.swapaxes(x, axis1=axis1, axis2=axis2)


def take(x, indices, axis=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices)
    if axis is None:
        x = mx.flatten(x)
        return mx.take(x, indices, axis=0)
    return mx.take(x, indices, axis=axis)


def take_along_axis(x, indices, axis=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices)
    if axis is None:
        x = mx.flatten(x)
        indices = mx.flatten(indices)
        return mx.take_along_axis(x, indices, axis=0)
    return mx.take_along_axis(x, indices, axis=axis)


def tan(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.tan(x)


def tanh(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "int64":
        dtype = config.floatx()
    else:
        dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.tanh(x)


def tensordot(x1, x2, axes=2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
    # mx.tensordot is backed by matmul and only supports float types.
    compute_dtype = dtypes.result_type(result_dtype, float)
    x1 = x1.astype(_to_mlx_dtype(compute_dtype))
    x2 = x2.astype(_to_mlx_dtype(compute_dtype))
    # mx.tensordot accepts an int or a `list[Sequence[int]]`; numpy also
    # accepts axes=(0, 1) and a pair of sequences as a tuple.
    if isinstance(axes, (list, tuple)):
        first, second = axes
        if not isinstance(first, (list, tuple)):
            first = (first,)
        if not isinstance(second, (list, tuple)):
            second = (second,)
        axes = [list(first), list(second)]
    return mx.tensordot(x1, x2, axes=axes).astype(_to_mlx_dtype(result_dtype))


def round(x, decimals=0):
    x = convert_to_tensor(x)
    return mx.round(x, decimals=decimals)


def tile(x, repeats):
    x = convert_to_tensor(x)
    return mx.tile(x, repeats)


def trace(x, offset=0, axis1=0, axis2=1):
    axis1 = standardize_axis_for_numpy(axis1)
    axis2 = standardize_axis_for_numpy(axis2)
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    # mx.trace doesn't support offset/axis args directly in all versions
    # Use diagonal + sum
    d = mx.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)
    return mx.sum(d, axis=-1).astype(_to_mlx_dtype(dtype))


def tri(N, M=None, k=0, dtype=None):
    dtype = dtype or config.floatx()
    if M is None:
        M = N
    return mx.tri(N, M, k, dtype=_to_mlx_dtype(dtype))


def tril(x, k=0):
    x = convert_to_tensor(x)
    return mx.tril(x, k=k)


def triu(x, k=0):
    x = convert_to_tensor(x)
    return mx.triu(x, k=k)


def trunc(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or "bool" == dtype:
        return x
    # trunc: round towards zero
    return mx.where(x >= 0, mx.floor(x), mx.ceil(x))


def vdot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    # mx.inner only supports float types; for integer/bool result dtypes
    # compute in float and cast back.
    compute_dtype = dtypes.result_type(dtype, float)
    x1 = mx.flatten(x1).astype(_to_mlx_dtype(compute_dtype))
    x2 = mx.flatten(x2).astype(_to_mlx_dtype(compute_dtype))
    return mx.inner(x1, x2).astype(_to_mlx_dtype(dtype))


def unique(
    x,
    sorted=True,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    size=None,
    fill_value=None,
):
    x = convert_to_tensor(x)
    is_flatten = axis is None
    original_shape = x.shape

    if is_flatten:
        dim = 0
        flat = mx.reshape(x, [-1])
        n = flat.shape[0]
        s_idx = mx.argsort(flat).astype(mx.int32)
        s = mx.take(flat, s_idx, axis=0)
        if n == 0:
            is_first = mx.zeros((0,), dtype=mx.bool_)
        else:
            is_first = mx.concatenate(
                [mx.array([True]), s[1:] != s[:-1]], axis=0
            )
    else:
        ndim = x.ndim
        dim = axis + ndim if axis < 0 else axis
        moved = mx.moveaxis(x, dim, 0)
        num_rows = moved.shape[0]
        num_cols = math.prod(moved.shape[1:])
        rows = mx.reshape(moved, (num_rows, num_cols))
        n = num_rows
        # Lexicographic (radix) sort: stable argsort per column, last to first.
        s_idx = mx.arange(num_rows, dtype=mx.int32)
        for col in range(num_cols - 1, -1, -1):
            ordered_col = mx.take(rows[:, col], s_idx, axis=0)
            perm = mx.argsort(ordered_col).astype(mx.int32)
            s_idx = mx.take(s_idx, perm, axis=0)
        s_rows = mx.take(rows, s_idx, axis=0)
        if num_rows == 0:
            is_first = mx.zeros((0,), dtype=mx.bool_)
        elif num_cols == 0:
            is_first = mx.concatenate(
                [mx.array([True]), mx.zeros((num_rows - 1,), dtype=mx.bool_)],
                axis=0,
            )
        else:
            adjacent_eq = mx.all(s_rows[1:] == s_rows[:-1], axis=1)
            is_first = mx.concatenate(
                [mx.array([True]), mx.logical_not(adjacent_eq)], axis=0
            )

    if n == 0:
        n_unique = 0
    else:
        pos = mx.cumsum(is_first.astype(mx.int32)) - 1
        n_unique = int(pos[-1].item()) + 1
    if n == 0:
        pos = mx.zeros((0,), dtype=mx.int32)

    # Map each original element (or row) to its unique slot via the sort perm.
    inverse_unsorted = mx.zeros((n,), dtype=mx.int32)
    if n > 0:
        inverse_unsorted = inverse_unsorted.at[s_idx].add(pos)

    if is_flatten:
        if n == 0:
            y = mx.zeros((0,), dtype=x.dtype)
        else:
            # `pos` is non-decreasing, so the first member of each slot is the
            # smallest sorted position; gather the sorted value there.
            first_sorted = mx.full((n_unique,), n, dtype=mx.int32)
            first_sorted = first_sorted.at[pos].minimum(
                mx.arange(n, dtype=mx.int32)
            )
            y = mx.take(s, first_sorted, axis=0)
    else:
        if n == 0:
            y_rows = mx.zeros((0, rows.shape[1]), dtype=x.dtype)
        else:
            first_sorted = mx.full((n_unique,), n, dtype=mx.int32)
            first_sorted = first_sorted.at[pos].minimum(
                mx.arange(n, dtype=mx.int32)
            )
            y_rows = mx.take(s_rows, first_sorted, axis=0)

    if return_counts:
        if n == 0:
            counts = mx.zeros((0,), dtype=mx.int32)
        else:
            counts = mx.zeros((n_unique,), dtype=mx.int32)
            counts = counts.at[pos].add(mx.ones((n,), dtype=mx.int32))

    if return_index:
        # First occurrence in original order = min original index per slot.
        if n == 0:
            unique_indices = mx.zeros((0,), dtype=mx.int32)
        else:
            big = mx.array(n, dtype=mx.int32)
            unique_indices = mx.full((n_unique,), big, dtype=mx.int32)
            orig_idx = mx.arange(n, dtype=mx.int32)
            unique_indices = unique_indices.at[inverse_unsorted].minimum(
                orig_idx
            )

    if not sorted and n_unique > 0:
        # Reorder unique slots into first-occurrence order: sort slots by the
        # smallest original index belonging to each slot.
        first_per_slot = mx.full((n_unique,), n, dtype=mx.int32)
        first_per_slot = first_per_slot.at[inverse_unsorted].minimum(
            mx.arange(n, dtype=mx.int32)
        )
        new_order = mx.argsort(first_per_slot).astype(mx.int32)
        remap = mx.zeros((n_unique,), dtype=mx.int32)
        remap = remap.at[new_order].add(mx.arange(n_unique, dtype=mx.int32))
        if is_flatten:
            y = mx.take(y, new_order, axis=0)
        else:
            y_rows = mx.take(y_rows, new_order, axis=0)
        if return_counts:
            counts = mx.take(counts, new_order, axis=0)
        if return_index:
            unique_indices = mx.take(unique_indices, new_order, axis=0)
        inverse_unsorted = mx.take(remap, inverse_unsorted, axis=0)

    # Reconstruct values back into the requested layout.
    if is_flatten:
        y_out = y
        inverse = mx.reshape(inverse_unsorted, original_shape)
    else:
        y_rows_shaped = mx.reshape(
            y_rows, (y_rows.shape[0],) + tuple(moved.shape[1:])
        )
        y_out = mx.moveaxis(y_rows_shaped, 0, dim)
        inverse = inverse_unsorted

    if size is not None:
        trunc = builtins.min(n_unique, size)
        pad = builtins.max(0, size - n_unique)
        fill = 0 if fill_value is None else fill_value

        keep = mx.arange(trunc, dtype=mx.int32)
        y_out = mx.take(y_out, keep, axis=dim)
        if pad > 0:
            pad_shape = list(y_out.shape)
            pad_shape[dim] = pad
            pad_block = mx.full(pad_shape, fill, dtype=y_out.dtype)
            y_out = mx.concatenate([y_out, pad_block], axis=dim)

        if return_index:
            unique_indices = mx.take(unique_indices, keep, axis=0)
            if pad > 0:
                unique_indices = mx.concatenate(
                    [unique_indices, mx.ones((pad,), dtype=mx.int32)], axis=0
                )
        if return_counts:
            counts = mx.take(counts, keep, axis=0)
            if pad > 0:
                counts = mx.concatenate(
                    [counts, mx.zeros((pad,), dtype=mx.int32)], axis=0
                )

    results = [y_out]
    if return_index:
        results.append(unique_indices)
    if return_inverse:
        results.append(inverse)
    if return_counts:
        results.append(counts)

    return tuple(results) if len(results) > 1 else results[0]


def inner(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype)
    # mx.inner only supports float types; for integer/bool result dtypes
    # compute in float and cast back.
    compute_dtype = dtypes.result_type(dtype, float)
    x1 = x1.astype(_to_mlx_dtype(compute_dtype))
    x2 = x2.astype(_to_mlx_dtype(compute_dtype))
    return mx.inner(x1, x2).astype(_to_mlx_dtype(dtype))


def vstack(xs):
    dtype_set = set([getattr(x, "dtype", type(x)) for x in xs])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        xs = [convert_to_tensor(x).astype(_to_mlx_dtype(dtype)) for x in xs]
    else:
        xs = [convert_to_tensor(x) for x in xs]
    # vstack: stack along axis=0 after ensuring at least 2D
    processed = []
    for x in xs:
        if x.ndim == 0:
            x = mx.reshape(x, (1, 1))
        elif x.ndim == 1:
            x = mx.reshape(x, (1, x.shape[0]))
        processed.append(x)
    return mx.concatenate(processed, axis=0)


def vsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    if x.ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return split(x, indices_or_sections, axis=0)


def vectorize(pyfunc, *, excluded=None, signature=None):
    return np.vectorize(pyfunc, excluded=excluded, signature=signature)


def where(condition, x1=None, x2=None):
    condition = convert_to_tensor(condition)
    if x1 is not None and x2 is not None:
        if not isinstance(x1, (int, float)):
            x1 = convert_to_tensor(x1)
        if not isinstance(x2, (int, float)):
            x2 = convert_to_tensor(x2)
        dtype = dtypes.result_type(
            getattr(x1, "dtype", type(x1)),
            getattr(x2, "dtype", type(x2)),
        )
        x1 = convert_to_tensor(x1, dtype)
        x2 = convert_to_tensor(x2, dtype)
        return mx.where(condition, x1, x2)
    else:
        return nonzero(condition)


def divide(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
        float,
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.divide(x1, x2)


def divide_no_nan(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
        float,
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.where(
        x2 == 0, convert_to_tensor(0, dtype=dtype), mx.divide(x1, x2)
    )


def true_divide(x1, x2):
    return divide(x1, x2)


def power(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.power(x1, x2)


def negative(x):
    x = convert_to_tensor(x)
    return mx.negative(x)


def nextafter(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(x1.dtype, x2.dtype, float)
    x1 = x1.astype(_to_mlx_dtype(dtype))
    x2 = x2.astype(_to_mlx_dtype(dtype))
    # IEEE 754: view floats as integers, increment/decrement to get next rep
    int_dtype = {2: mx.int16, 4: mx.int32}[x1.dtype.size]
    bits = x1.view(int_dtype)
    is_neg = bits < 0
    # Positive floats: +1 moves toward +inf; negative floats: -1 moves
    # toward -inf
    step_up = mx.where(is_neg, bits - 1, bits + 1)
    step_down = mx.where(is_neg, bits + 1, bits - 1)
    # -0 (sign bit only) stepping up → smallest positive (+1 as int)
    sign_bit = mx.array(1 << (x1.dtype.size * 8 - 1), dtype=int_dtype)
    step_up = mx.where(bits == sign_bit, mx.array(1, dtype=int_dtype), step_up)
    # +0 stepping down → smallest negative (-0 + 1 = 0x80000001)
    step_down = mx.where(
        bits == 0, mx.array(sign_bit | mx.array(1, dtype=int_dtype)), step_down
    )
    result_bits = mx.where(x2 > x1, step_up, mx.where(x2 < x1, step_down, bits))
    return result_bits.view(x1.dtype)


def square(x):
    x = convert_to_tensor(x)
    if standardize_dtype(x.dtype) == "bool":
        x = x.astype(mx.int32)
    return mx.square(x)


def sqrt(x):
    x = convert_to_tensor(x)
    dtype = (
        config.floatx()
        if standardize_dtype(x.dtype) == "int64"
        else dtypes.result_type(x.dtype, float)
    )
    x = x.astype(_to_mlx_dtype(dtype))
    return mx.sqrt(x)


def squeeze(x, axis=None):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    return mx.squeeze(x, axis=axis)


def transpose(x, axes=None):
    axes = tuple(axes) if isinstance(axes, list) else axes
    x = convert_to_tensor(x)
    return mx.transpose(x, axes=axes)


def trapezoid(y, x=None, dx=1.0, axis=-1):
    y = convert_to_tensor(y)
    result_dtype = dtypes.result_type(y.dtype, float)
    y = y.astype(_to_mlx_dtype(result_dtype))

    nd = y.ndim
    ax = axis
    if ax < 0:
        ax += nd

    slices_a = [builtins.slice(None)] * nd
    slices_b = [builtins.slice(None)] * nd
    slices_a[ax] = builtins.slice(1, None)
    slices_b[ax] = builtins.slice(None, -1)
    y_avg = (y[tuple(slices_a)] + y[tuple(slices_b)]) / 2.0

    if x is not None:
        x = convert_to_tensor(x).astype(_to_mlx_dtype(result_dtype))
        if x.ndim == 1:
            dx_arr = diff(x)
            # Broadcast dx_arr along the integration axis
            shape = [1] * nd
            shape[ax] = dx_arr.shape[0]
            dx_arr = mx.reshape(dx_arr, shape)
        else:
            slices_xa = [builtins.slice(None)] * x.ndim
            slices_xb = [builtins.slice(None)] * x.ndim
            slices_xa[ax] = builtins.slice(1, None)
            slices_xb[ax] = builtins.slice(None, -1)
            dx_arr = x[tuple(slices_xa)] - x[tuple(slices_xb)]
        return mx.sum(y_avg * dx_arr, axis=ax)

    dx = convert_to_tensor(dx, dtype=result_dtype)
    return mx.sum(y_avg, axis=ax) * dx


def vander(x, N=None, increasing=False):
    x = convert_to_tensor(x)
    result_dtype = dtypes.result_type(x.dtype)
    compute_dtype = dtypes.result_type(x.dtype, config.floatx())
    x = x.astype(_to_mlx_dtype(compute_dtype))
    if N is None:
        N = x.shape[0]
    if N == 0:
        return mx.zeros((x.shape[0], 0), dtype=_to_mlx_dtype(result_dtype))
    powers = mx.arange(N, dtype=_to_mlx_dtype(compute_dtype))
    if not increasing:
        powers = _flip(powers)
    # x[:, None] ** powers[None, :]
    result = mx.power(
        mx.expand_dims(x, axis=1),
        mx.expand_dims(powers, axis=0),
    )
    return result.astype(_to_mlx_dtype(result_dtype))


def var(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    compute_dtype = dtypes.result_type(x.dtype, "float32")
    result_dtype = dtypes.result_type(x.dtype, float)
    x = x.astype(_to_mlx_dtype(compute_dtype))
    return mx.var(x, axis=axis, keepdims=keepdims).astype(
        _to_mlx_dtype(result_dtype)
    )


def sum(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if dtype in ("bool", "int8", "int16"):
        dtype = "int32"
    elif dtype in ("uint8", "uint16"):
        dtype = "uint32"
    return mx.sum(x, axis=axis, keepdims=keepdims).astype(_to_mlx_dtype(dtype))


def eye(N, M=None, k=0, dtype=None):
    # The ops-layer float guard misses mlx float tensors because mlx Dtype
    # objects don't compare equal to their string names; standardize_dtype
    # does, so re-check here.
    for name, v in (("N", N), ("M", M)):
        v_dtype = getattr(v, "dtype", None)
        if (
            v is not None
            and not isinstance(v, int)
            and v_dtype is not None
            and standardize_dtype(v_dtype) in dtypes.FLOAT_TYPES
        ):
            raise TypeError(
                f"Argument `{name}` must be an integer or an integer tensor."
            )
    dtype = dtype or config.floatx()
    return mx.eye(N, m=M, k=k, dtype=_to_mlx_dtype(dtype))


def floor_divide(x1, x2):
    if not isinstance(x1, (int, float)):
        x1 = convert_to_tensor(x1)
    if not isinstance(x2, (int, float)):
        x2 = convert_to_tensor(x2)
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.floor_divide(x1, x2)


def logical_xor(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return mx.logical_and(
        mx.logical_or(x1, x2),
        mx.logical_not(mx.logical_and(x1, x2)),
    )


def corrcoef(x):
    x = convert_to_tensor(x)
    dtype_str = standardize_dtype(x.dtype)
    if dtype_str in ["int64", "float64"]:
        dtype = "float64"
    elif dtype_str in ["bfloat16", "float16"]:
        dtype = dtype_str
    else:
        dtype = config.floatx()
    x = x.astype(_to_mlx_dtype(dtype))
    # Covariance matrix → correlation matrix
    mean = mx.mean(x, axis=1, keepdims=True)
    xm = x - mean
    n = x.shape[1]
    cov = mx.matmul(xm, mx.transpose(xm)) / (n - 1)
    d = mx.sqrt(mx.diagonal(cov))
    outer_d = mx.expand_dims(d, 1) * mx.expand_dims(d, 0)
    return cov / outer_d


def correlate(x1, x2, mode="valid"):
    dtype = dtypes.result_type(
        getattr(x1, "dtype", type(x1)),
        getattr(x2, "dtype", type(x2)),
    )
    if dtype == "int64":
        dtype = "float64"
    elif dtype not in ["bfloat16", "float16", "float64"]:
        dtype = "float32"
    x1 = convert_to_tensor(x1, dtype)
    x2 = convert_to_tensor(x2, dtype)
    return mx.convolve(x1, _flip(x2), mode=mode)


def select(condlist, choicelist, default=0):
    condlist = [convert_to_tensor(c) for c in condlist]
    choicelist = [convert_to_tensor(c) for c in choicelist]
    result = mx.full(
        choicelist[0].shape,
        default,
        dtype=choicelist[0].dtype,
    )
    # Apply conditions in reverse order (last condition wins for ties)
    for cond, choice in reversed(list(zip(condlist, choicelist))):
        result = mx.where(cond, choice, result)
    return result


def slogdet(x):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    out_dtype = _to_mlx_dtype(dtype)
    n = x.shape[-1]
    # `mx.linalg.lu_factor` aborts the process (uncatchable C++ exception) on a
    # singular matrix, so detect singularity first via SVD (which does not
    # abort). Singular matrices are replaced with the identity before
    # `lu_factor` and their results overwritten with sign=0, logabsdet=-inf,
    # matching np.linalg.slogdet.
    s = mx.linalg.svd(x, compute_uv=False, stream=mx.cpu)
    s_max = mx.max(s, axis=-1)
    s_min = mx.min(s, axis=-1)
    tol = s_max * n * mx.finfo(x.dtype).eps
    singular = s_min <= tol
    eye = mx.eye(n, dtype=x.dtype)
    if x.ndim > 2:
        mask = singular.reshape(singular.shape + (1, 1))
    else:
        mask = singular.reshape((1, 1))
    x_safe = mx.where(mask, eye, x)
    lu, pivots = mx.linalg.lu_factor(x_safe, stream=mx.cpu)
    # Diagonal of LU gives the product for the determinant
    diag = mx.diagonal(lu, axis1=-2, axis2=-1)
    log_abs_det = mx.sum(mx.log(mx.abs(diag)), axis=-1)
    # Sign from diagonal signs
    sign_diag = mx.prod(mx.sign(diag), axis=-1)
    # Count permutation parity from pivots
    idx = mx.arange(n, dtype=pivots.dtype)
    n_swaps = mx.sum((pivots != idx).astype(mx.int32), axis=-1)
    parity = mx.where(n_swaps % 2 == 0, 1.0, -1.0)
    sign = sign_diag * parity
    sign = mx.where(singular, mx.zeros_like(sign), sign)
    neg_inf = mx.full(log_abs_det.shape, -mx.inf, dtype=log_abs_det.dtype)
    log_abs_det = mx.where(singular, neg_inf, log_abs_det)
    return sign.astype(out_dtype), log_abs_det.astype(out_dtype)


def argpartition(x, kth, axis=-1):
    x = convert_to_tensor(x)
    return mx.argpartition(x, kth, axis=axis).astype(mx.int32)


def histogram(x, bins=10, range=None):
    x = convert_to_tensor(x)
    x_flat = mx.flatten(x)
    if range is not None:
        lo, hi = float(range[0]), float(range[1])
    else:
        mx.eval(x_flat)
        lo, hi = float(mx.min(x_flat)), float(mx.max(x_flat))
    bin_edges = mx.linspace(lo, hi, bins + 1)
    # Assign each element to a bin via broadcast comparison
    # x_flat[:, None] >= bin_edges[None, :-1] and
    # x_flat[:, None] < bin_edges[None, 1:]
    x_exp = mx.expand_dims(x_flat, 1)
    edges_lo = mx.expand_dims(bin_edges[:-1], 0)
    edges_hi = mx.expand_dims(bin_edges[1:], 0)
    in_bin = mx.logical_and(x_exp >= edges_lo, x_exp < edges_hi)
    # Last bin is inclusive on the right
    in_last = mx.logical_and(
        x_exp >= edges_lo[:, -1:], x_exp <= edges_hi[:, -1:]
    )
    in_bin = mx.concatenate([in_bin[:, :-1], in_last], axis=1)
    hist = mx.sum(in_bin.astype(mx.int32), axis=0)
    return hist, bin_edges


def dsplit(x, indices_or_sections):
    x = convert_to_tensor(x)
    if x.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return split(x, indices_or_sections, axis=2)
