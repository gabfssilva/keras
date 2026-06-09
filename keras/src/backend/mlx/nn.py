import math

import mlx.core as mx

from keras.src import backend
from keras.src.backend.common.backend_utils import (
    compute_adaptive_pooling_window_sizes,
)
from keras.src.backend.common.backend_utils import (
    compute_conv_transpose_padding_args_for_jax,
)
from keras.src.backend.config import floatx
from keras.src.backend.mlx.core import _flip
from keras.src.backend.mlx.core import cast
from keras.src.backend.mlx.core import convert_to_tensor


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


def relu(x):
    x = convert_to_tensor(x)
    return mx.maximum(x, mx.array(0.0, dtype=x.dtype))


def relu6(x):
    x = convert_to_tensor(x)
    return mx.minimum(mx.maximum(x, mx.array(0.0, dtype=x.dtype)),
                      mx.array(6.0, dtype=x.dtype))


def sigmoid(x):
    x = convert_to_tensor(x)
    return mx.sigmoid(x)


def sparse_sigmoid(x):
    x = convert_to_tensor(x)
    return mx.where(
        x <= -1,
        mx.array(0.0, dtype=x.dtype),
        mx.where(
            x >= 1,
            mx.array(1.0, dtype=x.dtype),
            mx.array(0.5, dtype=x.dtype) * (x + mx.array(1.0, dtype=x.dtype)),
        ),
    )


def tanh(x):
    x = convert_to_tensor(x)
    return mx.tanh(x)


def tanh_shrink(x):
    x = convert_to_tensor(x)
    return x - mx.tanh(x)


def softplus(x):
    x = convert_to_tensor(x)
    return mx.logaddexp(x, mx.array(0.0, dtype=x.dtype))


def softsign(x):
    x = convert_to_tensor(x)
    return x / (mx.array(1.0, dtype=x.dtype) + mx.abs(x))


def soft_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return mx.where(
        x > threshold,
        x - mx.array(threshold, dtype=x.dtype),
        mx.where(
            x < -threshold,
            x + mx.array(threshold, dtype=x.dtype),
            mx.array(0.0, dtype=x.dtype),
        ),
    )


def sparse_plus(x):
    x = convert_to_tensor(x)
    return mx.where(
        x <= -1,
        mx.zeros_like(x),
        mx.where(
            x < 1,
            mx.array(0.25, dtype=x.dtype) * mx.square(x + 1),
            x,
        ),
    )


def silu(x):
    x = convert_to_tensor(x)
    return x * mx.sigmoid(x)


def squareplus(x, b=4):
    x = convert_to_tensor(x)
    b = convert_to_tensor(b, dtype=x.dtype)
    y = x + mx.sqrt(mx.square(x) + b)
    return y / mx.array(2.0, dtype=x.dtype)


def log_sigmoid(x):
    x = convert_to_tensor(x)
    return -softplus(-x)


def leaky_relu(x, negative_slope=0.2):
    x = convert_to_tensor(x)
    return mx.maximum(x, mx.array(negative_slope, dtype=x.dtype) * x)


def hard_sigmoid(x):
    x = convert_to_tensor(x)
    x = x / mx.array(6.0, dtype=x.dtype) + mx.array(0.5, dtype=x.dtype)
    return mx.where(
        x <= 0.0,
        mx.array(0.0, dtype=x.dtype),
        mx.where(x >= 1.0, mx.array(1.0, dtype=x.dtype), x),
    )


def hard_silu(x):
    x = convert_to_tensor(x)
    return x * hard_sigmoid(x)


def elu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return mx.where(
        x >= mx.array(0.0, dtype=x.dtype),
        x,
        mx.array(alpha, dtype=x.dtype) * mx.expm1(x),
    )


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    x = convert_to_tensor(x)
    return mx.array(scale, dtype=x.dtype) * elu(x, alpha)


def gelu(x, approximate=True):
    x = convert_to_tensor(x)
    if approximate:
        sqrt_2_over_pi = mx.array(
            math.sqrt(2.0 / math.pi), dtype=x.dtype
        )
        cdf = mx.array(0.5, dtype=x.dtype) * (
            mx.array(1.0, dtype=x.dtype)
            + mx.tanh(
                sqrt_2_over_pi
                * (x + mx.array(0.044715, dtype=x.dtype) * (x ** 3))
            )
        )
        return x * cdf
    else:
        sqrt_2 = mx.array(math.sqrt(2.0), dtype=x.dtype)
        return (
            x
            * (mx.erf(x / sqrt_2) + mx.array(1.0, dtype=x.dtype))
            / mx.array(2.0, dtype=x.dtype)
        )


def celu(x, alpha=1.0):
    x = convert_to_tensor(x)
    alpha = mx.array(alpha, dtype=x.dtype)
    return mx.maximum(x, mx.array(0.0, dtype=x.dtype)) + alpha * mx.expm1(
        mx.minimum(x, mx.array(0.0, dtype=x.dtype)) / alpha
    )


def glu(x, axis=-1):
    x = convert_to_tensor(x)
    dtype = x.dtype
    if x.shape[axis] % 2 != 0:
        raise ValueError(
            "axis size must be divisible by 2. "
            f"Received: x.shape={x.shape} with axis={axis}"
        )
    x1, x2 = mx.split(x, 2, axis=axis)
    return (x1 * mx.sigmoid(x2)).astype(dtype)


def hard_tanh(x):
    x = convert_to_tensor(x)
    return mx.clip(x, mx.array(-1.0, dtype=x.dtype),
                   mx.array(1.0, dtype=x.dtype))


def hard_shrink(x, threshold=0.5):
    x = convert_to_tensor(x)
    return mx.where(
        mx.abs(x) > mx.array(threshold, dtype=x.dtype),
        x,
        mx.array(0.0, dtype=x.dtype),
    )


def threshold(x, threshold, default_value):
    x = convert_to_tensor(x)
    return mx.where(
        x > threshold,
        x,
        mx.array(default_value, dtype=x.dtype),
    )


def softmax(x, axis=-1):
    x = convert_to_tensor(x)
    return mx.softmax(x, axis=axis)


def log_softmax(x, axis=-1):
    x = convert_to_tensor(x)
    max_x = mx.max(x, axis=axis, keepdims=True)
    shifted = x - max_x
    logsumexp = mx.log(mx.sum(mx.exp(shifted), axis=axis, keepdims=True))
    return shifted - logsumexp


def sparsemax(x, axis=-1):
    x = convert_to_tensor(x)
    # Sort in descending order along axis
    logits_sorted = -mx.sort(-x, axis=axis)
    logits_cumsum = mx.cumsum(logits_sorted, axis=axis)
    r = mx.arange(1, x.shape[axis] + 1)
    r_shape = [1] * x.ndim
    r_shape[axis] = -1
    r = mx.reshape(r, r_shape).astype(x.dtype)
    support = logits_sorted - (logits_cumsum - 1) / r > 0
    k = mx.sum(support.astype(mx.int32), axis=axis, keepdims=True).astype(
        x.dtype
    )
    logits_cumsum_safe = mx.where(support, logits_cumsum, mx.array(0.0, dtype=x.dtype))
    tau = (mx.sum(logits_cumsum_safe, axis=axis, keepdims=True) - 1) / k
    return mx.maximum(x - tau, mx.array(0.0, dtype=x.dtype))


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------


def _compute_same_padding(input_size, pool_size, stride):
    """Compute 'same' padding for a single spatial dimension."""
    out_size = math.ceil(input_size / stride)
    pad_total = max((out_size - 1) * stride + pool_size - input_size, 0)
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return pad_before, pad_after


def _gather_sum(inputs, pool_size, strides, num_spatial_dims):
    """Sum over sliding windows (used for counting valid entries)."""
    spatial_shape = inputs.shape[1 : 1 + num_spatial_dims]
    out_shape = []
    for i in range(num_spatial_dims):
        out_dim = (spatial_shape[i] - pool_size[i]) // strides[i] + 1
        out_shape.append(out_dim)

    batch_size = inputs.shape[0]
    channels = inputs.shape[-1]

    if num_spatial_dims == 1:
        out_l = out_shape[0]
        base_idx = mx.arange(out_l) * strides[0]
        result = mx.zeros(
            (batch_size, out_l, channels), dtype=inputs.dtype
        )
        for ki in range(pool_size[0]):
            result = result + inputs[:, base_idx + ki, :]

    elif num_spatial_dims == 2:
        out_h, out_w = out_shape
        base_h = mx.arange(out_h) * strides[0]
        base_w = mx.arange(out_w) * strides[1]
        result = mx.zeros(
            (batch_size, out_h, out_w, channels), dtype=inputs.dtype
        )
        for ki in range(pool_size[0]):
            for kj in range(pool_size[1]):
                result = result + inputs[:, base_h + ki][:, :, base_w + kj]

    elif num_spatial_dims == 3:
        out_d, out_h, out_w = out_shape
        base_d = mx.arange(out_d) * strides[0]
        base_h = mx.arange(out_h) * strides[1]
        base_w = mx.arange(out_w) * strides[2]
        result = mx.zeros(
            (batch_size, out_d, out_h, out_w, channels), dtype=inputs.dtype
        )
        for ki in range(pool_size[0]):
            for kj in range(pool_size[1]):
                for kk in range(pool_size[2]):
                    result = result + inputs[:, base_d + ki][
                        :, :, base_h + kj
                    ][:, :, :, base_w + kk]
    else:
        raise ValueError(
            f"Pooling not supported for {num_spatial_dims}D."
        )

    return result


def _pool_nd(inputs, pool_size, strides, padding, mode, data_format):
    """General N-D pooling using sliding-window gather + reduce."""
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2

    if isinstance(pool_size, int):
        pool_size = (pool_size,) * num_spatial_dims
    if isinstance(strides, int):
        strides = (strides,) * num_spatial_dims

    # Move to channels_last if needed
    if data_format == "channels_first":
        # (N, C, *spatial) -> (N, *spatial, C)
        perm = (0,) + tuple(range(2, 2 + num_spatial_dims)) + (1,)
        inputs = mx.transpose(inputs, perm)

    spatial_shape = inputs.shape[1 : 1 + num_spatial_dims]
    original_spatial_shape = spatial_shape
    pad_widths_saved = None

    # Handle padding
    if padding == "same":
        pad_widths = [(0, 0)]  # batch dim
        for i in range(num_spatial_dims):
            pb, pa = _compute_same_padding(
                spatial_shape[i], pool_size[i], strides[i]
            )
            pad_widths.append((pb, pa))
        pad_widths.append((0, 0))  # channel dim
        pad_widths_saved = pad_widths
        if mode == "max":
            pad_val = float("-inf")
        else:
            pad_val = 0.0
        inputs = mx.pad(inputs, pad_widths, constant_values=pad_val)
        spatial_shape = inputs.shape[1 : 1 + num_spatial_dims]

    # Compute output spatial dims
    out_shape = []
    for i in range(num_spatial_dims):
        out_dim = (spatial_shape[i] - pool_size[i]) // strides[i] + 1
        out_shape.append(out_dim)

    batch_size = inputs.shape[0]
    channels = inputs.shape[-1]

    # Build gather indices and perform pooling
    # We'll iterate over the pool window and accumulate
    if num_spatial_dims == 1:
        out_l = out_shape[0]
        base_idx = mx.arange(out_l) * strides[0]
        # Gather all window elements
        if mode == "max":
            result = mx.full((batch_size, out_l, channels), float("-inf"),
                             dtype=inputs.dtype)
        else:
            result = mx.zeros((batch_size, out_l, channels), dtype=inputs.dtype)

        for ki in range(pool_size[0]):
            idx = base_idx + ki
            gathered = inputs[:, idx, :]
            if mode == "max":
                result = mx.maximum(result, gathered)
            else:
                result = result + gathered

    elif num_spatial_dims == 2:
        out_h, out_w = out_shape
        base_h = mx.arange(out_h) * strides[0]
        base_w = mx.arange(out_w) * strides[1]

        if mode == "max":
            result = mx.full(
                (batch_size, out_h, out_w, channels),
                float("-inf"),
                dtype=inputs.dtype,
            )
        else:
            result = mx.zeros(
                (batch_size, out_h, out_w, channels), dtype=inputs.dtype
            )

        for ki in range(pool_size[0]):
            for kj in range(pool_size[1]):
                h_idx = base_h + ki
                w_idx = base_w + kj
                gathered = inputs[:, h_idx][:, :, w_idx]
                if mode == "max":
                    result = mx.maximum(result, gathered)
                else:
                    result = result + gathered

    elif num_spatial_dims == 3:
        out_d, out_h, out_w = out_shape
        base_d = mx.arange(out_d) * strides[0]
        base_h = mx.arange(out_h) * strides[1]
        base_w = mx.arange(out_w) * strides[2]

        if mode == "max":
            result = mx.full(
                (batch_size, out_d, out_h, out_w, channels),
                float("-inf"),
                dtype=inputs.dtype,
            )
        else:
            result = mx.zeros(
                (batch_size, out_d, out_h, out_w, channels),
                dtype=inputs.dtype,
            )

        for ki in range(pool_size[0]):
            for kj in range(pool_size[1]):
                for kk in range(pool_size[2]):
                    d_idx = base_d + ki
                    h_idx = base_h + kj
                    w_idx = base_w + kk
                    gathered = inputs[:, d_idx][:, :, h_idx][:, :, :, w_idx]
                    if mode == "max":
                        result = mx.maximum(result, gathered)
                    else:
                        result = result + gathered
    else:
        raise ValueError(
            f"Pooling is not supported for {num_spatial_dims}D inputs."
        )

    if mode == "average":
        pool_count = 1.0
        for ps in pool_size:
            pool_count *= ps
        if padding == "valid":
            result = result / mx.array(pool_count, dtype=result.dtype)
        else:
            # For 'same' padding, count valid (non-padded) entries per window.
            # Build a ones tensor with the original spatial shape, pad with
            # zeros, then do the same window sum to get per-window counts.
            # We use a single-element batch and single channel for efficiency.
            count_shape = [1] + list(original_spatial_shape) + [1]
            ones_count = mx.ones(count_shape, dtype=result.dtype)
            ones_padded = mx.pad(ones_count, pad_widths_saved)
            count_result = _gather_sum(
                ones_padded, pool_size, strides, num_spatial_dims
            )
            # Avoid division by zero
            count_result = mx.maximum(
                count_result, mx.array(1.0, dtype=result.dtype)
            )
            result = result / count_result

    # Move back to channels_first if needed
    if data_format == "channels_first":
        perm = (0, num_spatial_dims + 1) + tuple(range(1, num_spatial_dims + 1))
        result = mx.transpose(result, perm)

    return result



def max_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    inputs = convert_to_tensor(inputs)
    num_spatial_dims = inputs.ndim - 2
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * num_spatial_dims
    if strides is None:
        strides = pool_size
    return _pool_nd(inputs, pool_size, strides, padding, "max", data_format)


def average_pool(
    inputs,
    pool_size,
    strides=None,
    padding="valid",
    data_format=None,
):
    inputs = convert_to_tensor(inputs)
    num_spatial_dims = inputs.ndim - 2
    if isinstance(pool_size, int):
        pool_size = (pool_size,) * num_spatial_dims
    if strides is None:
        strides = pool_size
    return _pool_nd(
        inputs, pool_size, strides, padding, "average", data_format
    )


# ---------------------------------------------------------------------------
# Adaptive pooling
# ---------------------------------------------------------------------------


def _compute_adaptive_pooling_gather_indices(
    input_dim, output_size, big_window
):
    window_starts = mx.floor(
        (mx.arange(output_size).astype(mx.float32) * input_dim) / output_size
    ).astype(mx.int32)

    window_ends = mx.ceil(
        (mx.arange(1, output_size + 1).astype(mx.float32) * input_dim)
        / output_size
    ).astype(mx.int32)

    window_sizes = window_ends - window_starts
    is_big = window_sizes == big_window

    small_window = big_window - 1
    small_pool_len = input_dim - small_window + 1

    small_indices = window_starts
    big_indices = window_starts + small_pool_len

    return mx.where(is_big, big_indices, small_indices).astype(mx.int32)


def _strided_view_1d(x, window_size):
    """Extract sliding windows of `window_size` along axis=1.

    x: shape (N, L, C) -> output: (N, L - window_size + 1, window_size, C)
    """
    n, l, c = x.shape
    out_len = l - window_size + 1
    # Build index: (out_len, window_size)
    base = mx.arange(out_len).reshape(out_len, 1)
    offsets = mx.arange(window_size).reshape(1, window_size)
    indices = base + offsets  # (out_len, window_size)
    # Gather: x[:, indices, :] -> (N, out_len, window_size, C)
    return x[:, indices, :]


def _adaptive_pool1d_impl(inputs, output_size, mode, data_format):
    if isinstance(output_size, int):
        output_size = (output_size,)

    if data_format == "channels_first":
        inputs = mx.transpose(inputs, (0, 2, 1))

    n, l, c = inputs.shape
    out_l = output_size[0]

    small, big = compute_adaptive_pooling_window_sizes(l, out_l)
    gather = _compute_adaptive_pooling_gather_indices(l, out_l, big)

    sv_small = _strided_view_1d(inputs, small)
    small_pool = (
        mx.mean(sv_small, axis=2)
        if mode == "average"
        else mx.max(sv_small, axis=2)
    )

    sv_big = _strided_view_1d(inputs, big)
    big_pool = (
        mx.mean(sv_big, axis=2)
        if mode == "average"
        else mx.max(sv_big, axis=2)
    )

    combined = mx.concatenate([small_pool, big_pool], axis=1)
    out = combined[:, gather, :]

    if data_format == "channels_first":
        out = mx.transpose(out, (0, 2, 1))

    return out


def _adaptive_pool2d_impl(inputs, output_size, mode, data_format):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    if data_format == "channels_first":
        inputs = mx.transpose(inputs, (0, 2, 3, 1))

    n, h, w, c = inputs.shape
    out_h, out_w = output_size

    # Pool along height
    small_h, big_h = compute_adaptive_pooling_window_sizes(h, out_h)
    gather_h = _compute_adaptive_pooling_gather_indices(h, out_h, big_h)

    x_h = mx.transpose(inputs, (0, 2, 1, 3)).reshape(n * w, h, c)

    sv_small_h = _strided_view_1d(x_h, small_h)
    small_pool_h = (
        mx.mean(sv_small_h, axis=2)
        if mode == "average"
        else mx.max(sv_small_h, axis=2)
    )

    sv_big_h = _strided_view_1d(x_h, big_h)
    big_pool_h = (
        mx.mean(sv_big_h, axis=2)
        if mode == "average"
        else mx.max(sv_big_h, axis=2)
    )

    combined_h = mx.concatenate([small_pool_h, big_pool_h], axis=1)
    pooled_h = combined_h[:, gather_h, :]

    pooled_h = pooled_h.reshape(n, w, out_h, c)
    pooled_h = mx.transpose(pooled_h, (0, 2, 1, 3))

    # Pool along width
    small_w, big_w = compute_adaptive_pooling_window_sizes(w, out_w)
    gather_w = _compute_adaptive_pooling_gather_indices(w, out_w, big_w)

    x_w = pooled_h.reshape(n * out_h, w, c)

    sv_small_w = _strided_view_1d(x_w, small_w)
    small_pool_w = (
        mx.mean(sv_small_w, axis=2)
        if mode == "average"
        else mx.max(sv_small_w, axis=2)
    )

    sv_big_w = _strided_view_1d(x_w, big_w)
    big_pool_w = (
        mx.mean(sv_big_w, axis=2)
        if mode == "average"
        else mx.max(sv_big_w, axis=2)
    )

    combined_w = mx.concatenate([small_pool_w, big_pool_w], axis=1)
    out = combined_w[:, gather_w, :].reshape(n, out_h, out_w, c)

    if data_format == "channels_first":
        out = mx.transpose(out, (0, 3, 1, 2))

    return out


def _adaptive_pool3d_impl(inputs, output_size, mode, data_format):
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    if data_format == "channels_first":
        inputs = mx.transpose(inputs, (0, 2, 3, 4, 1))

    n, d, h, w, c = inputs.shape
    out_d, out_h, out_w = output_size

    # Pool along depth
    small_d, big_d = compute_adaptive_pooling_window_sizes(d, out_d)
    gather_d = _compute_adaptive_pooling_gather_indices(d, out_d, big_d)

    x_d = mx.transpose(inputs, (0, 2, 3, 1, 4)).reshape(n * h * w, d, c)

    sv_small_d = _strided_view_1d(x_d, small_d)
    small_pool_d = (
        mx.mean(sv_small_d, axis=2)
        if mode == "average"
        else mx.max(sv_small_d, axis=2)
    )

    sv_big_d = _strided_view_1d(x_d, big_d)
    big_pool_d = (
        mx.mean(sv_big_d, axis=2)
        if mode == "average"
        else mx.max(sv_big_d, axis=2)
    )

    combined_d = mx.concatenate([small_pool_d, big_pool_d], axis=1)
    pooled_d = combined_d[:, gather_d, :].reshape(n, h, w, out_d, c)
    pooled_d = mx.transpose(pooled_d, (0, 3, 1, 2, 4))

    # Pool along height
    small_h, big_h = compute_adaptive_pooling_window_sizes(h, out_h)
    gather_h = _compute_adaptive_pooling_gather_indices(h, out_h, big_h)

    x_h = mx.transpose(pooled_d, (0, 1, 3, 2, 4)).reshape(
        n * out_d * w, h, c
    )

    sv_small_h = _strided_view_1d(x_h, small_h)
    small_pool_h = (
        mx.mean(sv_small_h, axis=2)
        if mode == "average"
        else mx.max(sv_small_h, axis=2)
    )

    sv_big_h = _strided_view_1d(x_h, big_h)
    big_pool_h = (
        mx.mean(sv_big_h, axis=2)
        if mode == "average"
        else mx.max(sv_big_h, axis=2)
    )

    combined_h = mx.concatenate([small_pool_h, big_pool_h], axis=1)
    pooled_h = combined_h[:, gather_h, :].reshape(n, out_d, w, out_h, c)
    pooled_h = mx.transpose(pooled_h, (0, 1, 3, 2, 4))

    # Pool along width
    small_w, big_w = compute_adaptive_pooling_window_sizes(w, out_w)
    gather_w = _compute_adaptive_pooling_gather_indices(w, out_w, big_w)

    x_w = pooled_h.reshape(n * out_d * out_h, w, c)

    sv_small_w = _strided_view_1d(x_w, small_w)
    small_pool_w = (
        mx.mean(sv_small_w, axis=2)
        if mode == "average"
        else mx.max(sv_small_w, axis=2)
    )

    sv_big_w = _strided_view_1d(x_w, big_w)
    big_pool_w = (
        mx.mean(sv_big_w, axis=2)
        if mode == "average"
        else mx.max(sv_big_w, axis=2)
    )

    combined_w = mx.concatenate([small_pool_w, big_pool_w], axis=1)
    out = combined_w[:, gather_w, :].reshape(n, out_d, out_h, out_w, c)

    if data_format == "channels_first":
        out = mx.transpose(out, (0, 4, 1, 2, 3))

    return out


def adaptive_average_pool(inputs, output_size, data_format=None):
    inputs = convert_to_tensor(inputs)
    data_format = backend.standardize_data_format(data_format)
    dims = inputs.ndim - 2
    if dims == 1:
        return _adaptive_pool1d_impl(
            inputs, output_size, "average", data_format
        )
    if dims == 2:
        return _adaptive_pool2d_impl(
            inputs, output_size, "average", data_format
        )
    if dims == 3:
        return _adaptive_pool3d_impl(
            inputs, output_size, "average", data_format
        )
    raise ValueError("adaptive_average_pool supports only 1D/2D/3D")


def adaptive_max_pool(inputs, output_size, data_format=None):
    inputs = convert_to_tensor(inputs)
    data_format = backend.standardize_data_format(data_format)
    dims = inputs.ndim - 2
    if dims == 1:
        return _adaptive_pool1d_impl(inputs, output_size, "max", data_format)
    if dims == 2:
        return _adaptive_pool2d_impl(inputs, output_size, "max", data_format)
    if dims == 3:
        return _adaptive_pool3d_impl(inputs, output_size, "max", data_format)
    raise ValueError("adaptive_max_pool supports only 1D/2D/3D")


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------


def _convert_to_spatial_operand(x, num_spatial_dims):
    """Normalize a scalar or sequence to a tuple of length num_spatial_dims."""
    if isinstance(x, int):
        return (x,) * num_spatial_dims
    return tuple(x)


def _transpose_kernel_to_mlx(kernel, num_spatial_dims):
    """Transpose Keras kernel shape to MLX kernel shape.

    Keras: (*spatial, in_ch, out_ch)
    MLX:   (out_ch, *spatial, in_ch)
    """
    # axes: last -> 0, then spatial dims, then second-to-last
    ndim = kernel.ndim
    # spatial axes in Keras order: 0..num_spatial_dims-1
    # in_ch axis: num_spatial_dims
    # out_ch axis: num_spatial_dims + 1
    out_ch_axis = ndim - 1
    in_ch_axis = ndim - 2
    spatial_axes = tuple(range(num_spatial_dims))
    new_axes = (out_ch_axis,) + spatial_axes + (in_ch_axis,)
    return mx.transpose(kernel, new_axes)


def _compute_same_padding_conv(input_spatial, kernel_spatial, strides,
                                dilation_rate):
    """Compute explicit 'same' padding for convolution."""
    padding = []
    for i in range(len(input_spatial)):
        effective_kernel = (kernel_spatial[i] - 1) * dilation_rate[i] + 1
        out_size = math.ceil(input_spatial[i] / strides[i])
        pad_total = max(
            (out_size - 1) * strides[i] + effective_kernel - input_spatial[i],
            0,
        )
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        padding.append((pad_before, pad_after))
    return padding


def conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2

    strides = _convert_to_spatial_operand(strides, num_spatial_dims)
    dilation_rate = _convert_to_spatial_operand(dilation_rate, num_spatial_dims)

    # Determine channels and feature_group_count
    if data_format == "channels_last":
        channels = inputs.shape[-1]
    else:
        channels = inputs.shape[1]
    kernel_in_channels = kernel.shape[-2]
    if channels % kernel_in_channels > 0:
        raise ValueError(
            "The number of input channels must be evenly divisible by "
            f"kernel's in_channels. Received input channels {channels} and "
            f"kernel in_channels {kernel_in_channels}. "
        )
    feature_group_count = channels // kernel_in_channels

    # Convert to channels_last if needed (MLX expects channels_last)
    if data_format == "channels_first":
        perm = (0,) + tuple(range(2, 2 + num_spatial_dims)) + (1,)
        inputs = mx.transpose(inputs, perm)

    # Get spatial dimensions for padding computation
    input_spatial = inputs.shape[1 : 1 + num_spatial_dims]
    kernel_spatial = kernel.shape[:num_spatial_dims]

    # Transpose kernel from Keras format to MLX format
    mlx_kernel = _transpose_kernel_to_mlx(kernel, num_spatial_dims)

    # Handle padding
    if isinstance(padding, str) and padding.lower() == "same":
        pad_pairs = _compute_same_padding_conv(
            input_spatial, kernel_spatial, strides, dilation_rate
        )
        # Check if symmetric
        symmetric = all(p[0] == p[1] for p in pad_pairs)
        if symmetric:
            mlx_padding = [p[0] for p in pad_pairs]
        else:
            # Manually pad the input
            pad_widths = [(0, 0)]  # batch
            for p in pad_pairs:
                pad_widths.append(p)
            pad_widths.append((0, 0))  # channels
            inputs = mx.pad(inputs, pad_widths)
            mlx_padding = [0] * num_spatial_dims
    elif isinstance(padding, str) and padding.lower() == "valid":
        mlx_padding = [0] * num_spatial_dims
    else:
        # padding is already explicit (list of tuples)
        # Manually pad
        pad_widths = [(0, 0)]  # batch
        for p in padding:
            if isinstance(p, (list, tuple)):
                pad_widths.append(tuple(p))
            else:
                pad_widths.append((p, p))
        pad_widths.append((0, 0))  # channels
        inputs = mx.pad(inputs, pad_widths)
        mlx_padding = [0] * num_spatial_dims

    # Use conv_general for all dimensionalities
    if num_spatial_dims == 1:
        mlx_padding_val = mlx_padding[0]
        mlx_strides = strides[0]
        mlx_dilation = dilation_rate[0]
    else:
        mlx_padding_val = mlx_padding
        mlx_strides = strides
        mlx_dilation = dilation_rate

    result = mx.conv_general(
        inputs,
        mlx_kernel,
        stride=mlx_strides,
        padding=mlx_padding_val,
        kernel_dilation=mlx_dilation,
        groups=feature_group_count,
    )

    if result.size == 0 and inputs.size != 0:
        raise ValueError(
            "The convolution operation resulted in an empty output. "
            "This can happen if the input is too small for the given "
            "kernel size, strides, dilation rate, and padding mode. "
            "Please check the input shape and convolution parameters."
        )

    # Convert back to channels_first if needed
    if data_format == "channels_first":
        perm = (0, num_spatial_dims + 1) + tuple(
            range(1, num_spatial_dims + 1)
        )
        result = mx.transpose(result, perm)

    return result


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2

    # For depthwise conv, we reshape the kernel so that in_channels=1
    # and out_channels = in_channels * depth_multiplier, then use
    # groups = in_channels.
    if data_format == "channels_last":
        feature_group_count = inputs.shape[-1]
    else:
        feature_group_count = inputs.shape[1]

    # Keras depthwise kernel: (*spatial, in_ch, depth_mult)
    # We need to reshape to (*spatial, 1, in_ch * depth_mult)
    depth_multiplier = kernel.shape[-1]
    new_kernel_shape = (
        kernel.shape[:num_spatial_dims]
        + (1, feature_group_count * depth_multiplier)
    )
    kernel = mx.reshape(
        convert_to_tensor(kernel), new_kernel_shape
    )

    return conv(
        inputs,
        kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
    )


def separable_conv(
    inputs,
    depthwise_kernel,
    pointwise_kernel,
    strides=1,
    padding="valid",
    data_format=None,
    dilation_rate=1,
):
    data_format = backend.standardize_data_format(data_format)
    depthwise_conv_output = depthwise_conv(
        inputs,
        depthwise_kernel,
        strides,
        padding,
        data_format,
        dilation_rate,
    )
    return conv(
        depthwise_conv_output,
        pointwise_kernel,
        strides=1,
        padding="valid",
        data_format=data_format,
        dilation_rate=dilation_rate,
    )


def conv_transpose(
    inputs,
    kernel,
    strides=1,
    padding="valid",
    output_padding=None,
    data_format=None,
    dilation_rate=1,
):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)
    data_format = backend.standardize_data_format(data_format)
    num_spatial_dims = inputs.ndim - 2

    strides = _convert_to_spatial_operand(strides, num_spatial_dims)
    dilation_rate = _convert_to_spatial_operand(dilation_rate, num_spatial_dims)

    # Compute JAX-style padding pairs, which we can use with conv_general
    padding_values = compute_conv_transpose_padding_args_for_jax(
        input_shape=inputs.shape,
        kernel_shape=kernel.shape,
        strides=strides,
        padding=padding,
        output_padding=output_padding,
        dilation_rate=dilation_rate,
    )

    # Convert to channels_last if needed
    if data_format == "channels_first":
        perm = (0,) + tuple(range(2, 2 + num_spatial_dims)) + (1,)
        inputs = mx.transpose(inputs, perm)

    # The conv_transpose Keras kernel is (*spatial, out_ch, in_ch): the last
    # two axes are swapped relative to the forward conv. Map out_ch (the
    # second-to-last axis) and in_ch (the last axis) into the MLX
    # (out_ch, *spatial, in_ch) layout so C_in matches the input channels.
    ndim = kernel.ndim
    mlx_kernel = mx.transpose(
        kernel,
        (ndim - 2,) + tuple(range(num_spatial_dims)) + (ndim - 1,),
    )

    # Use conv_general with input_dilation for transposed convolution
    # padding_values is a list of (left_pad, right_pad) tuples
    # For conv_general, we need to compute the padding that achieves the
    # desired output shape. With input_dilation = strides and kernel_dilation
    # = dilation_rate, we use the JAX padding values.
    # conv_general padding format: tuple of (list_of_pad_before, list_of_pad_after)
    pad_before = [p[0] for p in padding_values]
    pad_after = [p[1] for p in padding_values]

    result = mx.conv_general(
        inputs,
        mlx_kernel,
        stride=1,
        padding=(pad_before, pad_after),
        kernel_dilation=dilation_rate,
        input_dilation=strides,
        flip=True,
    )

    # Convert back to channels_first if needed
    if data_format == "channels_first":
        perm = (0, num_spatial_dims + 1) + tuple(
            range(1, num_spatial_dims + 1)
        )
        result = mx.transpose(result, perm)

    return result


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    if sparse:
        raise ValueError(
            "Unsupported value `sparse=True` with mlx backend"
        )
    if dtype is None:
        dtype = "float32"
    x = convert_to_tensor(x)
    input_shape = x.shape

    x = mx.reshape(x, (-1,))
    if not num_classes:
        num_classes = int(mx.max(x).item()) + 1

    batch_size = x.shape[0]
    categorical = mx.zeros((batch_size, num_classes), dtype=mx.float32)
    valid_mask = x >= 0
    valid_indices = mx.arange(batch_size)

    # Use scatter-add approach
    categorical = categorical.at[valid_indices, x].add(
        mx.where(valid_mask, mx.array(1.0), mx.array(0.0))
    )

    # Reshape to output shape
    output_shape = input_shape + (num_classes,)
    categorical = mx.reshape(categorical, output_shape)

    # Move the new dimension to the right place
    if axis != -1:
        categorical = mx.moveaxis(categorical, -1, axis)

    from keras.src.backend.mlx.core import _to_mlx_dtype

    return categorical.astype(_to_mlx_dtype(dtype))


def multi_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    if sparse:
        raise ValueError(
            "Unsupported value `sparse=True` with mlx backend"
        )
    x = convert_to_tensor(x)
    reduction_axis = 1 if len(x.shape) > 1 else 0
    outputs = mx.max(
        one_hot(cast(x, "int32"), num_classes, axis=axis, dtype=dtype),
        axis=reduction_axis,
    )
    return outputs


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if len(target.shape) < 1:
        raise ValueError(
            "Arguments `target` and `output` must be at least rank 1. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        log_prob = log_softmax(output, axis=axis)
    else:
        output = output / mx.sum(output, axis=axis, keepdims=True)
        output = mx.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = mx.log(output)
    return -mx.sum(target * log_prob, axis=axis)


def sparse_categorical_crossentropy(
    target, output, from_logits=False, axis=-1
):
    target = convert_to_tensor(target, dtype="int32")
    output = convert_to_tensor(output)
    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = mx.squeeze(target, axis=-1)

    if len(output.shape) < 1:
        raise ValueError(
            "Argument `output` must be at least rank 1. "
            "Received: "
            f"output.shape={output.shape}"
        )
    if target.shape != output.shape[:-1]:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape "
            "up until the last dimension: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )
    if from_logits:
        log_prob = log_softmax(output, axis=axis)
    else:
        output = output / mx.sum(output, axis=axis, keepdims=True)
        output = mx.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = mx.log(output)
    target = one_hot(target, output.shape[axis], axis=axis)
    return -mx.sum(target * log_prob, axis=axis)


def binary_crossentropy(target, output, from_logits=False):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)

    if target.shape != output.shape:
        raise ValueError(
            "Arguments `target` and `output` must have the same shape. "
            "Received: "
            f"target.shape={target.shape}, output.shape={output.shape}"
        )

    if from_logits:
        output = mx.sigmoid(output)

    output = mx.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
    bce = target * mx.log(output)
    bce = bce + (1.0 - target) * mx.log(1.0 - output)
    return -bce


def ctc_loss(target, output, target_length, output_length, mask_index=0):
    target = convert_to_tensor(target, dtype="int32")
    output = convert_to_tensor(output)
    target_length = convert_to_tensor(target_length, dtype="int32")
    output_length = convert_to_tensor(output_length, dtype="int32")

    batch_size, max_input_length, num_classes = output.shape
    _, max_label_length = target.shape
    log_epsilon = -1e5

    dtype = backend.result_type(output.dtype, "float32")
    output = output.astype(dtype)

    def _lengths_to_paddings(lengths, max_length):
        indices = mx.arange(max_length).reshape(
            (1,) * lengths.ndim + (max_length,)
        )
        lengths = mx.expand_dims(lengths, axis=-1)
        elem_valid = indices < lengths
        return mx.logical_not(elem_valid)

    target_paddings = _lengths_to_paddings(
        target_length, max_label_length
    ).astype(output.dtype)
    output_paddings = _lengths_to_paddings(
        output_length, max_input_length
    ).astype(output.dtype)

    logprobs = log_softmax(output, axis=-1)
    label_lengths = max_label_length - mx.sum(
        target_paddings, axis=1
    ).astype(mx.int32)

    # repeat[b, n] == 1.0 when label[b, n] == label[b, n+1]
    repeat = (target[:, :-1] == target[:, 1:]).astype(mx.float32)
    repeat = mx.pad(repeat, ((0, 0), (0, 1)))

    logprobs_phi = logprobs[:, :, mask_index : mask_index + 1]  # [B, T, 1]
    logprobs_phi = mx.transpose(logprobs_phi, (1, 0, 2))  # [T, B, 1]

    _one_hot = one_hot(target, num_classes=num_classes)  # [B, N, K]
    logprobs_emit = mx.einsum("btk,bnk->btn", logprobs, _one_hot)
    logprobs_emit = mx.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

    logalpha_phi_init = (
        mx.ones((batch_size, max_label_length + 1), dtype=output.dtype)
        * log_epsilon
    )
    # Set first column to 0
    logalpha_phi_init = logalpha_phi_init.at[:, 0].add(
        mx.array(-log_epsilon, dtype=output.dtype)
    )
    logalpha_emit_init = (
        mx.ones((batch_size, max_label_length), dtype=output.dtype)
        * log_epsilon
    )

    def update_phi_score(phi, added_score):
        return mx.concatenate(
            [phi[:, :1], mx.logaddexp(phi[:, 1:], added_score)], axis=-1
        )

    def loop_body(prev, x):
        prev_phi, prev_emit = prev
        prev_phi_orig = prev_phi
        prev_phi = update_phi_score(
            prev_phi, prev_emit + log_epsilon * repeat
        )

        logprob_emit, logprob_phi, pad = x

        next_emit = mx.logaddexp(
            prev_phi[:, :-1] + logprob_emit, prev_emit + logprob_emit
        )
        next_phi = prev_phi + logprob_phi
        next_phi = update_phi_score(
            next_phi,
            prev_emit + logprob_phi + log_epsilon * (1.0 - repeat),
        )

        pad = pad.reshape((batch_size, 1))
        next_emit = pad * prev_emit + (1.0 - pad) * next_emit
        next_phi = pad * prev_phi_orig + (1.0 - pad) * next_phi

        return (next_phi, next_emit), (next_phi, next_emit)

    # Manual scan
    carry = (logalpha_phi_init, logalpha_emit_init)
    all_phi = []
    all_emit = []
    output_paddings_t = mx.transpose(output_paddings, (1, 0))
    for t in range(max_input_length):
        x = (logprobs_emit[t], logprobs_phi[t], output_paddings_t[t])
        carry, (phi_t, emit_t) = loop_body(carry, x)
        all_phi.append(phi_t)
        all_emit.append(emit_t)

    logalpha_phi = mx.stack(all_phi)
    logalpha_emit = mx.stack(all_emit)

    # Last row update with epsilon transition
    logalpha_phi_last = update_phi_score(
        logalpha_phi[-1], logalpha_emit[-1]
    )

    # Extract per_seq_loss
    _one_hot_labels = one_hot(
        label_lengths, num_classes=max_label_length + 1
    )
    per_seq_loss = -mx.sum(
        logalpha_phi_last * _one_hot_labels, axis=-1
    )
    return per_seq_loss


def _ctc_greedy_decode(
    inputs,
    sequence_lengths,
    merge_repeated=True,
    mask_index=None,
):
    inputs = convert_to_tensor(inputs)
    sequence_lengths = convert_to_tensor(sequence_lengths, dtype="int32")
    batch_size, max_length, num_classes = inputs.shape

    if mask_index is None:
        mask_index = num_classes - 1

    indices = mx.argmax(inputs, axis=-1).astype(mx.int32)
    scores = mx.max(inputs, axis=-1)

    seqlen_mask = mx.arange(max_length).reshape(1, -1)
    seqlen_mask = seqlen_mask >= mx.reshape(sequence_lengths, (-1, 1))

    indices = mx.where(seqlen_mask, mask_index, indices)
    scores = mx.where(seqlen_mask, mx.array(0.0, dtype=scores.dtype), scores)

    if merge_repeated:
        repeat_mask = indices[:, 1:] == indices[:, :-1]
        repeat_mask = mx.pad(repeat_mask, ((0, 0), (1, 0)))
        indices = mx.where(repeat_mask, mask_index, indices)

    # Set to -1 for blank labels
    invalid_mask = indices == mask_index
    indices = mx.where(invalid_mask, -1, indices)

    # Rearrange indices by moving blanks to the end
    order = mx.broadcast_to(
        mx.arange(max_length).reshape(1, -1), (batch_size, max_length)
    )
    order = mx.where(invalid_mask, max_length, order)
    order = mx.argsort(order, axis=-1)
    # Gather along last axis using advanced indexing
    batch_idx = mx.broadcast_to(
        mx.arange(batch_size).reshape(-1, 1), order.shape
    )
    indices = indices[batch_idx, order]

    scores = -mx.sum(scores, axis=1, keepdims=True)
    indices = mx.expand_dims(indices, axis=0)
    return indices, scores


def _ctc_beam_search_decode(
    inputs,
    sequence_lengths,
    beam_width=100,
    top_paths=1,
    mask_index=None,
):
    raise NotImplementedError(
        "Beam search CTC decoding is not supported with the MLX backend."
    )


def ctc_decode(
    inputs,
    sequence_lengths,
    strategy="greedy",
    beam_width=100,
    top_paths=1,
    merge_repeated=True,
    mask_index=0,
):
    inputs = convert_to_tensor(inputs)
    dtype = backend.result_type(inputs.dtype, "float32")
    inputs = cast(inputs, dtype)

    if strategy == "greedy":
        return _ctc_greedy_decode(
            inputs,
            sequence_lengths,
            merge_repeated=merge_repeated,
            mask_index=mask_index,
        )
    elif strategy == "beam_search":
        return _ctc_beam_search_decode(
            inputs,
            sequence_lengths,
            beam_width=beam_width,
            top_paths=top_paths,
            mask_index=mask_index,
        )
    else:
        raise ValueError(
            f"Invalid strategy {strategy}. Supported values are "
            "'greedy' and 'beam_search'."
        )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def moments(x, axes, keepdims=False, synchronized=False):
    if synchronized:
        raise NotImplementedError(
            "Argument synchronized=True is not supported with MLX."
        )
    x = convert_to_tensor(x)
    axes = tuple(axes) if isinstance(axes, list) else axes

    need_cast = False
    ori_dtype = backend.standardize_dtype(x.dtype)
    if ori_dtype == "float16":
        need_cast = True
        x = cast(x, "float32")

    mean = mx.mean(x, axis=axes, keepdims=True)
    # Var = E[|x|^2] - |E[x]|^2
    variance = mx.mean(mx.square(x), axis=axes, keepdims=True) - mx.square(
        mean
    )

    if not keepdims:
        mean = mx.squeeze(mean, axis=axes)
        variance = mx.squeeze(variance, axis=axes)
    if need_cast:
        mean = mx.clip(
            mean,
            mx.array(mx.finfo(mx.float16).min),
            mx.array(mx.finfo(mx.float16).max),
        )
        variance = mx.clip(
            variance,
            mx.array(mx.finfo(mx.float16).min),
            mx.array(mx.finfo(mx.float16).max),
        )
        mean = cast(mean, ori_dtype)
        variance = cast(variance, ori_dtype)
    return mean, variance


def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    x = convert_to_tensor(x)
    mean = convert_to_tensor(mean)
    variance = convert_to_tensor(variance)

    shape = [1] * len(x.shape)
    shape[axis] = mean.shape[0]
    mean = mx.reshape(mean, shape)
    variance = mx.reshape(variance, shape)

    inv = mx.array(1.0, dtype=x.dtype) / mx.sqrt(
        variance + mx.array(epsilon, dtype=variance.dtype)
    )
    if scale is not None:
        scale = convert_to_tensor(scale)
        scale = mx.reshape(scale, shape)
        inv = inv * scale

    res = -mean * inv
    if offset is not None:
        offset = convert_to_tensor(offset)
        offset = mx.reshape(offset, shape)
        res = res + offset

    return x * inv + res


# ---------------------------------------------------------------------------
# Image operations
# ---------------------------------------------------------------------------


def psnr(x1, x2, max_val):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    if x1.shape != x2.shape:
        raise ValueError(
            f"Input shapes {x1.shape} and {x2.shape} must "
            "match for PSNR calculation. "
        )

    max_val = convert_to_tensor(max_val, dtype=x2.dtype)
    mse = mx.mean(mx.square(x1 - x2))
    psnr_val = 20 * mx.log10(max_val) - 10 * mx.log10(mse)
    return psnr_val


def depth_to_space(x, block_size, data_format="channels_last"):
    x = convert_to_tensor(x)
    if data_format == "channels_last":
        n, h, w, c = x.shape
        new_c = c // (block_size ** 2)
        x = mx.reshape(x, (n, h, w, block_size, block_size, new_c))
        x = mx.transpose(x, (0, 1, 3, 2, 4, 5))
        x = mx.reshape(x, (n, h * block_size, w * block_size, new_c))
    else:
        n, c, h, w = x.shape
        new_c = c // (block_size ** 2)
        x = mx.reshape(x, (n, new_c, block_size, block_size, h, w))
        x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
        x = mx.reshape(x, (n, new_c, h * block_size, w * block_size))
    return x


def space_to_depth(x, block_size, data_format="channels_last"):
    x = convert_to_tensor(x)
    if data_format == "channels_last":
        n, h, w, c = x.shape
        new_h = h // block_size
        new_w = w // block_size
        x = mx.reshape(x, (n, new_h, block_size, new_w, block_size, c))
        x = mx.transpose(x, (0, 1, 3, 2, 4, 5))
        x = mx.reshape(x, (n, new_h, new_w, c * block_size ** 2))
    else:
        n, c, h, w = x.shape
        new_h = h // block_size
        new_w = w // block_size
        x = mx.reshape(x, (n, c, new_h, block_size, new_w, block_size))
        x = mx.transpose(x, (0, 1, 3, 5, 2, 4))
        x = mx.reshape(x, (n, c * block_size ** 2, new_h, new_w))
    return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def _get_large_negative(dtype):
    dtype = backend.standardize_dtype(dtype)
    val = 65500.0 if dtype == "float16" else 3.38953e38
    return mx.array(val * -0.7, dtype=dtype)


def _apply_masks(logits, mask, is_causal):
    if mask is None and not is_causal:
        return logits

    combined_mask = mx.ones(logits.shape, dtype=mx.bool_)
    if mask is not None:
        combined_mask = mx.logical_and(combined_mask, mask)

    if is_causal:
        T, S = logits.shape[2], logits.shape[3]
        causal_mask = mx.tril(mx.ones((T, S), dtype=mx.bool_))
        causal_mask = mx.reshape(causal_mask, (1, 1, T, S))
        combined_mask = mx.logical_and(combined_mask, causal_mask)

    padded_logits = mx.where(
        combined_mask, logits, _get_large_negative(logits.dtype)
    )
    return padded_logits


def dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    scale=None,
    is_causal=False,
    flash_attention=None,
    attn_logits_soft_cap=None,
):
    if flash_attention is None:
        flash_attention = False
    if flash_attention:
        raise ValueError(
            "Flash attention is not supported in mlx backend."
        )

    query = convert_to_tensor(query)
    key = convert_to_tensor(key)
    value = convert_to_tensor(value)
    if len(query.shape) != 4:
        raise ValueError(
            "`dot_product_attention` only supports 4D inputs. "
            f"Received: query.shape={query.shape}, key.shape={key.shape}, "
            f"value.shape={value.shape}."
        )
    compute_dtype = backend.result_type(query.dtype, key.dtype, value.dtype)
    query = cast(query, compute_dtype)
    key = cast(key, compute_dtype)
    value = cast(value, compute_dtype)
    if bias is not None:
        bias = convert_to_tensor(bias, dtype=compute_dtype)

    _, _, _, H = key.shape
    scale = (1.0 / math.sqrt(H)) if scale is None else scale

    # query: (B, T, N, H), key: (B, S, N, H) -> logits: (B, N, T, S)
    logits = mx.einsum("BTNH,BSNH->BNTS", query, key)
    logits = logits * mx.array(scale, dtype=logits.dtype)

    if bias is not None:
        logits = logits + bias

    if attn_logits_soft_cap is not None:
        logits = mx.tanh(logits / attn_logits_soft_cap) * attn_logits_soft_cap

    padded_logits = _apply_masks(logits, mask, is_causal)

    # Softmax in float32 for stability
    padded_logits = padded_logits.astype(mx.float32)
    probs = mx.softmax(padded_logits, axis=-1).astype(
        value.dtype
    )

    # probs: (B, N, T, S), value: (B, S, N, H) -> encoded: (B, T, N, H)
    encoded = mx.einsum("BNTS,BSNH->BTNH", probs, value)
    return encoded


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------


def dropout(inputs, rate, noise_shape=None, seed=None):
    from keras.src.backend.mlx import random as mlx_random

    return mlx_random.dropout(inputs, rate, noise_shape=noise_shape, seed=seed)


# ---------------------------------------------------------------------------
# Unfold / Fold
# ---------------------------------------------------------------------------


def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """Extract sliding local blocks from a NCHW batched image tensor.

    Args:
        input: 4-D tensor, shape (N, C, H, W).
        kernel_size: int or (kH, kW)
        dilation: int or (dH, dW), default 1
        padding: int or (pH, pW), default 0
        stride: int or (sH, sW), default 1

    Returns:
        3-D tensor, shape (N, C*kH*kW, L)
    """
    input = convert_to_tensor(input)

    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    k = _pair(kernel_size)
    d = _pair(dilation)
    p = _pair(padding)
    s = _pair(stride)

    N, C, H, W = input.shape

    # Padding
    if any(v > 0 for v in p):
        input = mx.pad(
            input,
            ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])),
        )

    # Output spatial size
    oH = (input.shape[2] - (k[0] - 1) * d[0] - 1) // s[0] + 1
    oW = (input.shape[3] - (k[1] - 1) * d[1] - 1) // s[1] + 1

    # Build gather indices
    i0 = mx.arange(0, oH) * s[0]
    j0 = mx.arange(0, oW) * s[1]

    # Collect patches
    patches = []
    for idx in range(k[0]):
        for jdx in range(k[1]):
            h_indices = i0 + idx * d[0]
            w_indices = j0 + jdx * d[1]
            # input[:, :, h_indices, :][:, :, :, w_indices]
            # -> (N, C, oH, oW)
            gathered = input[:, :, h_indices][:, :, :, w_indices]
            patches.append(mx.reshape(gathered, (N, C, -1)))

    # Stack: (N, C, k[0]*k[1], oH*oW) then reshape
    # Actually each patch is (N, C, oH*oW), we have k[0]*k[1] of them
    # We need shape (N, C*kH*kW, L)
    # Concatenate along axis=1
    result = mx.concatenate(patches, axis=1)
    # Currently (N, C * kH * kW, L) -- but we concatenated C-dim patches
    # Actually each patch is (N, C, oH*oW), concatenating k0*k1 of them
    # gives (N, C * k0 * k1, oH*oW) -- but that interleaves wrong.
    # We need to stack them as (N, k0*k1, C, oH*oW) then transpose to
    # (N, C, k0*k1, oH*oW) then reshape to (N, C*k0*k1, oH*oW)
    # Let me redo this properly.

    # Build as list of (N, C, oH*oW) arrays, one per kernel position
    # Stack to (k0*k1, N, C, L)
    stacked = mx.stack(patches, axis=0)  # (k0*k1, N, C, L)
    # Transpose to (N, C, k0*k1, L)
    stacked = mx.transpose(stacked, (1, 2, 0, 3))
    # Reshape to (N, C*k0*k1, L)
    L = oH * oW
    result = mx.reshape(stacked, (N, C * k[0] * k[1], L))

    return result


def fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """Combine an array of sliding local blocks into a large tensor (col2im).

    Args:
        x: 3-D tensor, shape (N, C*kH*kW, L).
        output_size: int or (oH, oW)
        kernel_size: int or (kH, kW)
        dilation: int or (dH, dW), default 1
        padding: int or (pH, pW), default 0
        stride: int or (sH, sW), default 1

    Returns:
        4-D tensor, shape (N, C, oH, oW)
    """
    x = convert_to_tensor(x)

    def _pair(val):
        return (val, val) if isinstance(val, int) else val

    oH, oW = _pair(output_size)
    kH, kW = _pair(kernel_size)
    dH, dW = _pair(dilation)
    pH, pW = _pair(padding)
    sH, sW = _pair(stride)

    N, CKK, L = x.shape
    C = CKK // (kH * kW)

    nH = (oH + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    nW = (oW + 2 * pW - dW * (kW - 1) - 1) // sW + 1

    # Reshape: (N, C*kH*kW, L) -> (N, C, kH, kW, nH, nW)
    x = mx.reshape(x, (N, C, kH, kW, nH, nW))

    # Padded output size
    oH_pad = oH + 2 * pH
    oW_pad = oW + 2 * pW

    # Flatten N*C for vectorized scatter
    # x shape: (N, C, kH, kW, nH, nW)
    # Reshape to (N*C, kH, kW, nH, nW) and output to (N*C, oH_pad, oW_pad)
    x_flat = mx.reshape(x, (N * C, kH, kW, nH, nW))
    output = mx.zeros((N * C, oH_pad, oW_pad), dtype=x.dtype)

    for i in range(kH):
        for j in range(kW):
            h_start = i * dH
            w_start = j * dW
            h_arr = mx.array(h_start) + mx.arange(nH) * sH
            w_arr = mx.array(w_start) + mx.arange(nW) * sW
            h_ix = mx.reshape(h_arr, (-1, 1))
            w_ix = mx.reshape(w_arr, (1, -1))
            h_grid = mx.flatten(mx.broadcast_to(h_ix, (nH, nW)))
            w_grid = mx.flatten(mx.broadcast_to(w_ix, (nH, nW)))

            # patch: (N*C, nH*nW)
            patch = mx.reshape(x_flat[:, i, j, :, :], (N * C, nH * nW))
            output = output.at[:, h_grid, w_grid].add(patch)

    output = mx.reshape(output, (N, C, oH_pad, oW_pad))

    # Remove padding
    if pH > 0 or pW > 0:
        output = output[:, :, pH : oH_pad - pH, pW : oW_pad - pW]

    return output
