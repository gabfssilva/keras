import builtins
import contextlib
import functools
import warnings

import mlx.core as mx

# np is ONLY for the I/O bridge (convert_to_tensor input,
# convert_to_numpy output)
import numpy as np

# All computation must use mlx.core — never np for math/indexing
from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.symbolic_scope import SymbolicScope


def _flip(x, axis=None):
    """Reverse elements of x along the given axis (pure MLX)."""
    if axis is None:
        # Flip all axes
        slices = tuple(builtins.slice(None, None, -1) for _ in range(x.ndim))
    else:
        if isinstance(axis, int):
            axis = (axis,)
        slices = tuple(
            builtins.slice(None, None, -1)
            if i in axis
            else builtins.slice(None)
            for i in range(x.ndim)
        )
    return x[slices]


SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
SUPPORTS_COMPLEX_DTYPES = True
IS_THREAD_SAFE = False

MLX_DTYPES = {
    "float16": mx.float16,
    "float32": mx.float32,
    "float64": mx.float32,  # MLX has no float64 — see _to_mlx_dtype warning
    "bfloat16": mx.bfloat16,
    "int8": mx.int8,
    "int16": mx.int16,
    "int32": mx.int32,
    "int64": mx.int64,
    "uint8": mx.uint8,
    "uint16": mx.uint16,
    "uint32": mx.uint32,
    "uint64": mx.uint64,
    "bool": mx.bool_,
    "complex64": mx.complex64,
    "complex128": mx.complex64,  # MLX has no complex128, downcast
}


_FLOAT64_WARNING_ISSUED = False
_COMPLEX128_WARNING_ISSUED = False


def _to_mlx_dtype(dtype):
    global _FLOAT64_WARNING_ISSUED
    global _COMPLEX128_WARNING_ISSUED
    dtype = standardize_dtype(dtype)
    if dtype == "float64" and not _FLOAT64_WARNING_ISSUED:
        _FLOAT64_WARNING_ISSUED = True
        warnings.warn(
            "MLX does not support float64. Downcasting to float32. "
            "This may cause precision loss in operations that depend "
            "on 64-bit precision (e.g. numerical gradients, large "
            "reductions). Set your default dtype to float32 to "
            "silence this warning.",
            stacklevel=3,
        )
    if dtype == "complex128" and not _COMPLEX128_WARNING_ISSUED:
        _COMPLEX128_WARNING_ISSUED = True
        warnings.warn(
            "MLX does not support complex128. Downcasting to complex64. "
            "This may cause precision loss in operations that depend on "
            "128-bit complex precision.",
            stacklevel=3,
        )
    if dtype in MLX_DTYPES:
        return MLX_DTYPES[dtype]
    raise ValueError(f"Unsupported dtype for MLX backend: {dtype}")


class Variable(KerasVariable):
    def _initialize(self, value):
        self._value = convert_to_tensor(value, dtype=self.dtype)

    def _direct_assign(self, value):
        self._value = convert_to_tensor(value, dtype=self.dtype)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    def __array__(self):
        return np.array(self.value)


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if sparse:
        raise ValueError("`sparse=True` is not supported with mlx backend")
    if ragged:
        raise ValueError("`ragged=True` is not supported with mlx backend")
    if dtype is not None:
        dtype = standardize_dtype(dtype)
    if isinstance(x, Variable):
        if dtype and dtype != x.dtype:
            return mx.array(np.array(x.value), dtype=_to_mlx_dtype(dtype))
        return x.value
    if isinstance(x, mx.array):
        if dtype is None:
            return x
        target_dtype = _to_mlx_dtype(dtype)
        if x.dtype == target_dtype:
            return x
        return x.astype(target_dtype)
    if dtype is None:
        dtype = result_type(
            *[getattr(item, "dtype", type(item)) for item in tree.flatten(x)]
        )
    mlx_dtype = _to_mlx_dtype(dtype)
    if isinstance(x, (list, tuple)) and any(
        isinstance(item, mx.array) for item in tree.flatten(x)
    ):
        # Avoid the numpy round-trip: it would force evaluation, which is
        # not allowed while tracing under `mx.compile`.
        return mx.stack([convert_to_tensor(item, dtype=dtype) for item in x])
    return mx.array(np.array(x, dtype=dtype), dtype=mlx_dtype)


def convert_to_numpy(x):
    if isinstance(x, Variable):
        x = x.value
    if isinstance(x, mx.array):
        mx.eval(x)
        return np.array(x)
    return np.array(x)


def is_tensor(x):
    return isinstance(x, mx.array)


def shape(x):
    return tuple(x.shape)


def cast(x, dtype):
    return convert_to_tensor(x, dtype=dtype)


def cond(pred, true_fn, false_fn):
    if pred:
        return true_fn()
    return false_fn()


def vectorized_map(function, elements):
    if not isinstance(elements, (list, tuple)):
        return mx.stack([function(x) for x in elements])
    else:
        batch_size = elements[0].shape[0]
        output_store = []
        for index in range(batch_size):
            output_store.append(function([x[index] for x in elements]))
        return mx.stack(output_store)


def compute_output_spec(fn, *args, **kwargs):
    with StatelessScope(), SymbolicScope():

        def has_none_shape(x):
            if isinstance(x, KerasTensor):
                return None in x.shape
            return False

        none_in_shape = any(
            builtins.map(has_none_shape, tree.flatten((args, kwargs)))
        )

        def convert_keras_tensor_to_mlx(x, fill_value=None):
            if isinstance(x, KerasTensor):
                s = list(x.shape)
                if fill_value:
                    for i, e in enumerate(s):
                        if e is None:
                            s[i] = fill_value
                return mx.zeros(s, dtype=_to_mlx_dtype(x.dtype))
            return x

        args_1, kwargs_1 = tree.map_structure(
            lambda x: convert_keras_tensor_to_mlx(x, fill_value=83),
            (args, kwargs),
        )
        outputs_1 = fn(*args_1, **kwargs_1)
        outputs = outputs_1

        if none_in_shape:
            args_2, kwargs_2 = tree.map_structure(
                lambda x: convert_keras_tensor_to_mlx(x, fill_value=89),
                (args, kwargs),
            )
            outputs_2 = fn(*args_2, **kwargs_2)

            flat_out_1 = tree.flatten(outputs_1)
            flat_out_2 = tree.flatten(outputs_2)
            flat_out = []
            for x1, x2 in zip(flat_out_1, flat_out_2):
                s = list(x1.shape)
                for i, e in enumerate(x2.shape):
                    if e != s[i]:
                        s[i] = None
                flat_out.append(KerasTensor(s, standardize_dtype(x1.dtype)))
            outputs = tree.pack_sequence_as(outputs_1, flat_out)

        def convert_mlx_to_keras_tensor(x):
            if is_tensor(x):
                return KerasTensor(x.shape, standardize_dtype(x.dtype))
            return x

        output_spec = tree.map_structure(convert_mlx_to_keras_tensor, outputs)
    return output_spec


def map(f, xs):
    def g(_, x):
        return (), f(x)

    _, ys = scan(g, (), xs)
    return ys


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    if not callable(f):
        raise TypeError(f"`f` should be a callable. Received: f={f}")
    if not isinstance(unroll, bool):
        if not isinstance(unroll, int) or unroll < 1:
            raise ValueError(
                "`unroll` must be an positive integer or boolean. "
                f"Received: unroll={unroll}"
            )
    if xs is None and length is None:
        raise ValueError("Got no `xs` to scan over and `length` not provided.")

    input_is_sequence = tree.is_nested(xs)
    output_is_sequence = tree.is_nested(init)

    def pack_input(x):
        return tree.pack_sequence_as(xs, x) if input_is_sequence else x[0]

    def pack_output(x):
        return tree.pack_sequence_as(init, x) if output_is_sequence else x[0]

    if xs is None:
        xs_flat = []
        n = int(length)
    else:
        xs_flat = tree.flatten(xs)
        xs_flat = [convert_to_tensor(elem) for elem in xs_flat]
        n = int(length) if length is not None else shape(xs_flat[0])[0]

    init_flat = tree.flatten(init)
    init_flat = [convert_to_tensor(init) for init in init_flat]
    init = pack_output(init_flat)
    dummy_y = [mx.zeros_like(init) for init in init_flat]

    carry = init
    ys = []
    maybe_reversed = reversed if reverse else lambda x: x
    for i in maybe_reversed(range(n)):
        xs_slice = [x[i] for x in xs_flat]
        packed_xs = pack_input(xs_slice) if len(xs_slice) > 0 else None
        carry, y = f(carry, packed_xs)
        ys.append(y if y is not None else dummy_y)
    stacked_y = tree.map_structure(
        lambda *ys: mx.stack(ys), *maybe_reversed(ys)
    )
    return carry, stacked_y


def associative_scan(f, elems, reverse=False, axis=0):
    # Pure MLX implementation — no numpy conversions
    elems_flat = tree.flatten(elems)
    elems_flat = [convert_to_tensor(elem) for elem in elems_flat]
    if reverse:
        elems_flat = [_flip(elem, axis=axis) for elem in elems_flat]

    def _combine(a_flat, b_flat):
        a = tree.pack_sequence_as(elems, a_flat)
        b = tree.pack_sequence_as(elems, b_flat)
        c = f(a, b)
        c_flat = tree.flatten(c)
        return c_flat

    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same "
            "first dimension. (saw: {})".format(
                [elem.shape for elem in elems_flat]
            )
        )

    def _slice(elem, s, step=1):
        """Take a slice along `axis` using pure MLX ops."""
        n = elem.shape[axis]
        start = s.start if s.start is not None else 0
        stop = s.stop if s.stop is not None else n
        sl_step = s.step if s.step is not None else step
        # Handle negative indices
        if start < 0:
            start = max(0, n + start)
        if stop < 0:
            stop = max(0, n + stop)
        indices = mx.arange(start, stop, sl_step)
        return mx.take(elem, indices, axis=axis)

    def _interleave(a, b, axis):
        if not (
            a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
        ):
            raise ValueError(
                "Shapes are incompatible for associative_scan "
                "interleaving. "
                f"a.shape[{axis}]={a.shape[axis]}, "
                f"b.shape[{axis}]={b.shape[axis]}"
            )
        # Dilate a: insert zeros between elements
        a_len = a.shape[axis]
        a_dil_len = a_len * 2 - 1
        # Build dilated arrays via indexing
        a_shape = list(a.shape)
        a_shape[axis] = a_dil_len
        a_dil = mx.zeros(a_shape, dtype=a.dtype)
        even_indices = mx.arange(0, a_dil_len, 2)
        a_dil = a_dil.at[
            tuple(
                builtins.slice(None) if i != axis else even_indices
                for i in range(a.ndim)
            )
        ].add(a)

        b_len = b.shape[axis]
        b_dil_len = b_len * 2 - 1
        b_shape = list(b.shape)
        b_shape[axis] = b_dil_len
        b_dil = mx.zeros(b_shape, dtype=b.dtype)
        b_even_indices = mx.arange(0, b_dil_len, 2)
        b_dil = b_dil.at[
            tuple(
                builtins.slice(None) if i != axis else b_even_indices
                for i in range(b.ndim)
            )
        ].add(b)

        # Pad and combine
        a_pad_widths = [(0, 0)] * a.ndim
        a_pad_widths[axis] = (0, 1 if a_len == b_len else 0)
        b_pad_widths = [(0, 0)] * b.ndim
        b_pad_widths[axis] = (1, 0) if a_len == b_len else (1, 1)
        a_padded = mx.pad(a_dil, a_pad_widths)
        b_padded = mx.pad(b_dil, b_pad_widths)
        if a.dtype == mx.bool_:
            return mx.logical_or(a_padded, b_padded)
        return a_padded + b_padded

    def _scan(elems):
        num_elems = elems[0].shape[axis]
        if num_elems < 2:
            return elems
        reduced_elems = _combine(
            [_slice(e, builtins.slice(0, -1, 2)) for e in elems],
            [_slice(e, builtins.slice(1, None, 2)) for e in elems],
        )
        odd_elems = _scan(reduced_elems)
        if num_elems % 2 == 0:
            even_elems = _combine(
                [_slice(e, builtins.slice(0, -1)) for e in odd_elems],
                [_slice(e, builtins.slice(2, None, 2)) for e in elems],
            )
        else:
            even_elems = _combine(
                odd_elems,
                [_slice(e, builtins.slice(2, None, 2)) for e in elems],
            )
        even_elems = [
            mx.concatenate(
                [_slice(elem, builtins.slice(0, 1)), result],
                axis=axis,
            )
            for (elem, result) in zip(elems, even_elems)
        ]
        return list(
            builtins.map(
                functools.partial(_interleave, axis=axis),
                even_elems,
                odd_elems,
            )
        )

    scans = _scan(elems_flat)
    if reverse:
        scans = [_flip(s, axis=axis) for s in scans]
    return tree.pack_sequence_as(elems, scans)


def scatter(indices, values, shape):
    indices = convert_to_tensor(indices)
    values = convert_to_tensor(values)
    index_length = indices.shape[-1]
    value_shape = shape[index_length:]
    flat_indices = mx.reshape(indices, (-1, index_length))
    flat_values = mx.reshape(values, [-1] + list(value_shape))
    # Decompose multi-dim indices into per-axis index arrays
    idx = tuple(flat_indices[:, i] for i in range(index_length))
    zeros = mx.zeros(shape, dtype=values.dtype)
    zeros = zeros.at[idx].add(flat_values)
    return zeros


def scatter_update(inputs, indices, updates, reduction=None):
    inputs = convert_to_tensor(inputs)
    indices = convert_to_tensor(indices)
    updates = convert_to_tensor(updates)
    indices_t = mx.transpose(indices)
    idx = tuple(indices_t[i] for i in range(indices_t.shape[0]))
    if reduction is None:
        # Direct assignment: zero out target, then add
        result = inputs.at[idx].add(updates - inputs[idx])
    elif reduction == "add":
        result = inputs.at[idx].add(updates)
    elif reduction == "max":
        result = inputs.at[idx].maximum(updates)
    elif reduction == "min":
        result = inputs.at[idx].minimum(updates)
    elif reduction == "mul":
        result = inputs.at[idx].multiply(updates)
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    return result


def slice(inputs, start_indices, shape):
    if len(start_indices) != len(shape):
        raise ValueError(
            "Length of `start_indices` must match length of `shape`. "
            f"Received: start_indices={start_indices}, shape={shape}"
        )
    inputs = convert_to_tensor(inputs)
    # Build slice using Python slicing (pure MLX)
    slices = tuple(
        builtins.slice(int(s), int(s) + int(sz))
        for s, sz in zip(start_indices, shape)
    )
    return inputs[slices]


def slice_update(inputs, start_indices, updates):
    inputs = convert_to_tensor(inputs)
    updates = convert_to_tensor(updates)
    start = mx.array([int(s) for s in start_indices], dtype=mx.int32)
    axes = list(range(len(start_indices)))
    return mx.slice_update(inputs, updates, start, axes)


def switch(index, branches, *operands):
    index = convert_to_tensor(index, "int32")
    index = int(index.item())
    index = max(0, min(index, len(branches) - 1))
    return branches[index](*operands)


def while_loop(cond, body, loop_vars, maximum_iterations=None):
    current_iter = 0
    iteration_check = lambda iter: (
        maximum_iterations is None or iter < maximum_iterations
    )
    is_tuple = isinstance(loop_vars, (tuple, list))
    loop_vars = tuple(loop_vars) if is_tuple else (loop_vars,)
    loop_vars = tree.map_structure(convert_to_tensor, loop_vars)
    while cond(*loop_vars) and iteration_check(current_iter):
        loop_vars = body(*loop_vars)
        if not isinstance(loop_vars, (list, tuple)):
            loop_vars = (loop_vars,)
        loop_vars = tuple(loop_vars)
        current_iter += 1
    return loop_vars if is_tuple else loop_vars[0]


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def stop_gradient(variable):
    return mx.stop_gradient(variable)


def unstack(x, num=None, axis=0):
    x = convert_to_tensor(x)
    # mx.split along axis, then squeeze
    sections = x.shape[axis]
    parts = mx.split(x, sections, axis=axis)
    return [mx.squeeze(p, axis=axis) for p in parts]


def random_seed_dtype():
    return "uint32"


class custom_gradient:
    """Wraps a function that returns (output, grad_fn) into an
    MLX-differentiable function using mx.custom_function.

    Keras convention: ``fun(*args) -> (output, grad_fn)`` where
    ``grad_fn(upstream) -> tuple[grad_per_arg, ...]``.

    MLX convention: ``@mx.custom_function`` with a ``.vjp`` method
    ``vjp(primals, cotangents, outputs) -> tuple[grad_per_primal, ...]``.
    """

    def __init__(self, fun):
        self.fun = fun

    def __call__(self, *args, **kwargs):
        # Capture the user's grad_fn during the forward pass so the
        # vjp can call it later.
        captured = {}

        @mx.custom_function
        def _forward(*primals):
            outputs, grad_fn = self.fun(*primals, **kwargs)
            captured["grad_fn"] = grad_fn
            # custom_function expects a single array or tuple of arrays
            if isinstance(outputs, (list, tuple)):
                return tuple(outputs)
            return (outputs,)

        @_forward.vjp
        def _vjp(primals, cotangents, outputs):
            grad_fn = captured["grad_fn"]
            # Keras grad_fn receives upstream gradient(s) and returns
            # one gradient per *primal* input.
            grads = grad_fn(*cotangents)
            if not isinstance(grads, (list, tuple)):
                grads = (grads,)
            return tuple(grads)

        result = _forward(*args)
        # Unwrap single-element tuple to match Keras convention
        if len(result) == 1:
            return result[0]
        return result


@contextlib.contextmanager
def device_scope(device_name):
    if device_name and "gpu" in device_name.lower():
        device = mx.gpu
    else:
        device = mx.cpu
    with mx.stream(device):
        yield


def remat(f):
    return mx.checkpoint(f)
