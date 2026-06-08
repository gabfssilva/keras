import math

import mlx.core as mx
import mlx.core.fft as mx_fft
import numpy as np

from keras.src.backend import standardize_dtype
from keras.src.backend.mlx.core import _flip
from keras.src.backend.mlx.core import convert_to_tensor


def segment_sum(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    if num_segments is None:
        num_segments = int(mx.max(segment_ids).item()) + 1

    # Filter out negative segment_ids
    valid_mask = segment_ids >= 0
    valid_data = data[valid_mask]
    valid_segment_ids = segment_ids[valid_mask]

    data_shape = list(valid_data.shape)
    data_shape[0] = num_segments

    result = mx.zeros(data_shape, dtype=valid_data.dtype)
    result = result.at[valid_segment_ids].add(valid_data)
    return result


def segment_max(data, segment_ids, num_segments=None, sorted=False):
    data = convert_to_tensor(data)
    segment_ids = convert_to_tensor(segment_ids)
    if num_segments is None:
        num_segments = int(mx.max(segment_ids).item()) + 1

    # Filter out negative segment_ids
    valid_mask = segment_ids >= 0
    valid_data = data[valid_mask]
    valid_segment_ids = segment_ids[valid_mask]

    data_shape = list(valid_data.shape)
    data_shape[0] = num_segments

    result = mx.full(data_shape, -mx.inf, dtype=valid_data.dtype)
    result = result.at[valid_segment_ids].maximum(valid_data)
    return result


def top_k(x, k, sorted=True):
    x = convert_to_tensor(x)
    if sorted:
        sorted_indices = mx.argsort(x, axis=-1)
        # Reverse to get descending order
        sorted_indices = _flip(sorted_indices, axis=-1)
        sorted_values = mx.take_along_axis(x, sorted_indices, axis=-1)
        top_k_values = sorted_values[..., :k]
        top_k_indices = sorted_indices[..., :k]
    else:
        # Use argpartition for unsorted top-k
        neg_x = -x
        top_k_indices = mx.argpartition(neg_x, kth=k - 1, axis=-1)[..., :k]
        top_k_values = mx.take_along_axis(x, top_k_indices, axis=-1)
    return top_k_values, top_k_indices


def in_top_k(targets, predictions, k):
    targets = convert_to_tensor(targets)
    predictions = convert_to_tensor(predictions)
    targets = mx.expand_dims(targets, axis=-1)
    topk_values = top_k(predictions, k)[0]
    targets_values = mx.take_along_axis(predictions, targets, axis=-1)
    mask = targets_values >= topk_values
    return mx.any(mask, axis=-1)


def logsumexp(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return mx.logsumexp(x, axis=axis, keepdims=keepdims)


def qr(x, mode="reduced"):
    x = convert_to_tensor(x)
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    # MLX's linalg.qr always returns the reduced factorisation.
    # For "complete" we fall back to numpy.
    if mode == "complete":
        x_np = np.array(x)
        q_np, r_np = np.linalg.qr(x_np, mode="complete")
        return convert_to_tensor(q_np), convert_to_tensor(r_np)
    return mx.linalg.qr(x)


def extract_sequences(x, sequence_length, sequence_stride):
    x = convert_to_tensor(x)
    *batch_shape, signal_length = x.shape
    batch_shape = list(batch_shape)
    num_sequences = (
        (signal_length - (sequence_length - sequence_stride))
        // sequence_stride
    )
    shape = x.shape[:-1] + (num_sequences, sequence_length)
    strides = x.strides[:-1] + (
        sequence_stride * x.strides[-1],
        x.strides[-1],
    )
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.reshape(x, (*batch_shape, *x.shape[-2:]))


def _get_complex_tensor_from_tuple(x):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            f"Received: x={x}"
        )
    real, imag = x
    real = convert_to_tensor(real)
    imag = convert_to_tensor(imag)
    if real.shape != imag.shape:
        raise ValueError(
            "Input `x` should be a tuple of two tensors - real and imaginary."
            "Both the real and imaginary parts should have the same shape. "
            f"Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}"
        )
    if not mx.issubdtype(real.dtype, mx.floating) or not mx.issubdtype(
        imag.dtype, mx.floating
    ):
        raise ValueError(
            "At least one tensor in input `x` is not of type float."
            f"Received: x={x}."
        )
    complex_input = real + 1j * imag
    return complex_input


def fft(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = mx_fft.fft(complex_input)
    return mx.real(complex_output), mx.imag(complex_output)


def fft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = mx_fft.fft2(complex_input)
    return mx.real(complex_output), mx.imag(complex_output)


def ifft2(x):
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = mx_fft.ifft2(complex_input)
    return mx.real(complex_output), mx.imag(complex_output)


def rfft(x, fft_length=None):
    x = convert_to_tensor(x)
    complex_output = mx_fft.rfft(x, n=fft_length, axis=-1)
    return mx.real(complex_output), mx.imag(complex_output)


def irfft(x, fft_length=None):
    complex_input = _get_complex_tensor_from_tuple(x)
    return mx_fft.irfft(complex_input, n=fft_length, axis=-1)


def _overlap_sequences(x, sequence_stride):
    """Overlap-add sequences along the last two dimensions.

    Ref: https://github.com/google/jax/blob/main/jax/_src/scipy/signal.py
    """
    x = convert_to_tensor(x)
    *batch_shape, num_sequences, sequence_length = x.shape

    if sequence_stride > sequence_length:
        raise ValueError(
            "`sequence_stride` must equal or less than x.shape[-1]. "
            f"Received: sequence_stride={sequence_stride}, "
            f"x.shape[-1]={sequence_length}"
        )
    if sequence_stride < (sequence_length / num_sequences):
        raise ValueError(
            "`sequence_stride` must equal or greater than "
            "x.shape[-1] / x.shape[-2]. "
            f"Received: sequence_stride={sequence_stride}, "
            f"x.shape[-1]={sequence_length}, x.shape[-2]={num_sequences}"
        )

    flat_batchsize = math.prod(batch_shape)
    x = mx.reshape(x, (flat_batchsize, num_sequences, sequence_length))
    output_size = sequence_stride * (num_sequences - 1) + sequence_length
    nstep_per_segment = 1 + (sequence_length - 1) // sequence_stride

    padded_segment_len = nstep_per_segment * sequence_stride
    pad_amount = padded_segment_len - sequence_length
    if pad_amount > 0:
        x = mx.pad(x, [(0, 0), (0, 0), (0, pad_amount)])

    x = mx.reshape(
        x, (flat_batchsize, num_sequences, nstep_per_segment, sequence_stride)
    )
    x = mx.transpose(x, (0, 2, 1, 3))  # (B, S, N, T)
    x = mx.pad(x, [(0, 0), (0, 0), (0, num_sequences), (0, 0)])

    shrinked = x.shape[2] - 1
    x = mx.reshape(x, (flat_batchsize, -1))
    x = x[:, : (nstep_per_segment * shrinked * sequence_stride)]
    x = mx.reshape(
        x, (flat_batchsize, nstep_per_segment, shrinked * sequence_stride)
    )
    x = mx.sum(x, axis=1)[:, :output_size]
    return mx.reshape(x, tuple(batch_shape) + (-1,))


def _get_window(window_name, sequence_length, dtype):
    """Generate a window tensor using native MLX window functions."""
    n = int(sequence_length)
    if window_name == "hann":
        win = 0.5 - 0.5 * mx.cos(
            2 * 3.141592653589793 * mx.arange(n).astype(mx.float32) / (n - 1)
        )
    elif window_name == "hamming":
        win = 0.54 - 0.46 * mx.cos(
            2 * 3.141592653589793 * mx.arange(n).astype(mx.float32) / (n - 1)
        )
    elif window_name == "blackman":
        a = mx.arange(n).astype(mx.float32) / (n - 1)
        win = (0.42 - 0.5 * mx.cos(2 * 3.141592653589793 * a)
               + 0.08 * mx.cos(4 * 3.141592653589793 * a))
    elif window_name == "bartlett":
        win = 1.0 - mx.abs(
            2.0 * mx.arange(n).astype(mx.float32) / (n - 1) - 1.0
        )
    else:
        raise ValueError(
            f"Unsupported window type: {window_name}. "
            "Supported: hann, hamming, blackman, bartlett."
        )
    return win.astype(dtype)


def stft(
    x, sequence_length, sequence_stride, fft_length, window="hann",
    center=True
):
    if standardize_dtype(x.dtype) not in {"float32", "float64"}:
        raise TypeError(
            "Invalid input type. Expected `float32` or `float64`. "
            f"Received: input type={x.dtype}"
        )
    if fft_length < sequence_length:
        raise ValueError(
            "`fft_length` must equal or larger than `sequence_length`. "
            f"Received: sequence_length={sequence_length}, "
            f"fft_length={fft_length}"
        )
    if isinstance(window, str):
        if window not in {"hann", "hamming"}:
            raise ValueError(
                "If a string is passed to `window`, it must be one of "
                f'`"hann"`, `"hamming"`. Received: window={window}'
            )
    x = convert_to_tensor(x)
    ori_dtype = x.dtype

    if center:
        # Reflect padding along the last axis
        pad_len = fft_length // 2
        # Reflect: x[..., pad:0:-1] | x | x[..., -2:-2-pad:-1]
        left = x[..., pad_len:0:-1]
        right = x[..., -2:-2 - pad_len:-1]
        x = mx.concatenate([left, x, right], axis=-1)

    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            win = _get_window(window, sequence_length, ori_dtype)
        else:
            win = convert_to_tensor(window, dtype=standardize_dtype(ori_dtype))
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        win = mx.pad(win, [(l_pad, r_pad)])
    else:
        win = mx.ones((sequence_length + l_pad + r_pad,), dtype=ori_dtype)

    # Extract overlapping sequences
    sequences = extract_sequences(x, fft_length, sequence_stride)
    # Apply window: sequences has shape (..., num_sequences, fft_length)
    sequences = sequences * win

    # Apply rfft along last axis
    complex_output = mx_fft.rfft(sequences, n=fft_length, axis=-1)

    # Scale
    scale = mx.sqrt(1.0 / mx.sum(win) ** 2)
    complex_output = complex_output / scale

    return (
        mx.real(complex_output).astype(ori_dtype),
        mx.imag(complex_output).astype(ori_dtype),
    )


def istft(
    x,
    sequence_length,
    sequence_stride,
    fft_length,
    length=None,
    window="hann",
    center=True,
):
    complex_input = _get_complex_tensor_from_tuple(x)
    dtype = mx.real(complex_input).dtype

    expected_output_len = fft_length + sequence_stride * (
        complex_input.shape[-2] - 1
    )
    l_pad = (fft_length - sequence_length) // 2
    r_pad = fft_length - sequence_length - l_pad

    if window is not None:
        if isinstance(window, str):
            win = _get_window(window, sequence_length, dtype)
        else:
            win = convert_to_tensor(window, dtype=standardize_dtype(dtype))
        if len(win.shape) != 1 or win.shape[-1] != sequence_length:
            raise ValueError(
                "The shape of `window` must be equal to [sequence_length]."
                f"Received: window shape={win.shape}"
            )
        win = mx.pad(win, [(l_pad, r_pad)])
    else:
        win = mx.ones((sequence_length + l_pad + r_pad,), dtype=dtype)

    # Inverse rfft: complex_input shape is (..., num_sequences, fft_bins)
    x = mx_fft.irfft(complex_input, n=fft_length, axis=-1)

    if window is not None:
        # Compute the squared-window normalisation denominator
        _sequence_length = sequence_length + l_pad + r_pad
        denom = mx.square(win)
        overlaps = -(-_sequence_length // sequence_stride)
        pad_amount = overlaps * sequence_stride - _sequence_length
        if pad_amount > 0:
            denom = mx.pad(denom, [(0, pad_amount)])
        denom = mx.reshape(denom, (overlaps, sequence_stride))
        denom = mx.sum(denom, axis=0, keepdims=True)
        denom = mx.tile(denom, (overlaps, 1))
        denom = mx.reshape(denom, (overlaps * sequence_stride,))
        win = win / denom[:_sequence_length]
        x = x * win
    else:
        x = x / sequence_stride

    x = _overlap_sequences(x, sequence_stride)

    start = 0 if center is False else fft_length // 2
    if length is not None:
        end = start + length
    elif center is True:
        end = -(fft_length // 2)
    else:
        end = expected_output_len
    return x[..., start:end]


def rsqrt(x):
    x = convert_to_tensor(x)
    return mx.rsqrt(x)


def erf(x):
    x = convert_to_tensor(x)
    return mx.erf(x)


def erfinv(x):
    x = convert_to_tensor(x)
    return mx.erfinv(x)


def logdet(x):
    x = convert_to_tensor(x)
    # Use slogdet for numerical stability
    sign, logabsdet = mx.linalg.slogdet(x)
    return logabsdet
