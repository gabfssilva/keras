import mlx.core as mx

from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.mlx.core import _to_mlx_dtype
from keras.src.backend.mlx.core import convert_to_tensor


def cholesky(a, upper=False):
    a = convert_to_tensor(a)
    # MLX linalg ops are CPU-only; force a CPU compute stream.
    out = mx.linalg.cholesky(a, upper=upper, stream=mx.cpu)
    # MLX silently returns a wrong factor for non-PSD inputs (no NaN, no
    # error), so detect it via the reconstruction. The op-level wrapper turns
    # any raised exception into a ValueError.
    factor = mx.transpose(out, _matrix_transpose_axes(a.ndim)) if upper else out
    reconstructed = mx.matmul(
        factor, mx.transpose(factor, _matrix_transpose_axes(a.ndim))
    )
    if not mx.allclose(reconstructed, a, rtol=1e-2, atol=1e-4):
        raise ValueError("Input is not a positive-definite matrix.")
    return out


def _matrix_transpose_axes(ndim):
    axes = list(range(ndim))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return axes


def cholesky_inverse(a, upper=False):
    a = convert_to_tensor(a)
    identity = mx.eye(a.shape[-1], dtype=a.dtype)
    inv_chol = solve_triangular(a, identity, lower=not upper)
    transpose_axes = _matrix_transpose_axes(inv_chol.ndim)
    # Run the final matmul on the CPU stream: the GPU float32 matmul of these
    # small triangular-inverse matrices is too inaccurate for the test atol.
    if upper:
        a_inv = mx.matmul(
            inv_chol, mx.transpose(inv_chol, transpose_axes), stream=mx.cpu
        )
    else:
        a_inv = mx.matmul(
            mx.transpose(inv_chol, transpose_axes), inv_chol, stream=mx.cpu
        )
    return a_inv


def det(a):
    # MLX does not have a native det. Compute via LU decomposition:
    # P @ L @ U = A, so det(A) = det(P) * det(L) * det(U).
    # det(L) = 1 (unit lower triangular), det(U) = prod(diag(U)),
    # det(P) = sign of the permutation encoded by P.
    a = convert_to_tensor(a)
    # MLX linalg ops are CPU-only; force a CPU compute stream.
    p, l, u = mx.linalg.lu(a, stream=mx.cpu)
    diag_u = mx.diagonal(u, axis1=-2, axis2=-1)
    det_u = mx.prod(diag_u, axis=-1)

    n = a.shape[-1]
    # MLX's `lu` returns P as 1D permutation indices, not a permutation matrix.
    p_indices = p

    if a.ndim == 2:
        sign = _permutation_sign(p_indices, n, det_u.dtype)
        return sign * det_u
    else:
        batch_shape = a.shape[:-2]
        flat_p = mx.reshape(p_indices, (-1, n))
        flat_det_u = mx.reshape(det_u, (-1,))
        signs = []
        for i in range(flat_p.shape[0]):
            signs.append(_permutation_sign(flat_p[i], n, flat_det_u.dtype))
        signs = mx.array(signs, dtype=flat_det_u.dtype)
        result = signs * flat_det_u
        return mx.reshape(result, batch_shape)


def _permutation_sign(perm, n, dtype=mx.float32):
    """Compute the sign of a permutation (+1 or -1).

    Uses cycle counting: sign = (-1)^(n - num_cycles).
    """
    visited = [False] * n
    num_cycles = 0
    perm_list = perm.tolist()
    for i in range(n):
        if not visited[i]:
            num_cycles += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = int(perm_list[j])
    swaps = n - num_cycles
    return 1.0 if swaps % 2 == 0 else -1.0


def eig(a):
    a = convert_to_tensor(a)
    # MLX linalg ops are CPU-only; force a CPU compute stream.
    return mx.linalg.eig(a, stream=mx.cpu)


def eigh(a):
    a = convert_to_tensor(a)
    # MLX linalg ops are CPU-only; force a CPU compute stream.
    return mx.linalg.eigh(a, stream=mx.cpu)


def inv(a):
    a = convert_to_tensor(a)
    # MLX linalg ops are CPU-only; force a CPU compute stream.
    return mx.linalg.inv(a, stream=mx.cpu)


def lu_factor(a):
    a = convert_to_tensor(a)
    # MLX linalg ops are CPU-only; force a CPU compute stream.
    return mx.linalg.lu_factor(a, stream=mx.cpu)


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, "float32")
        x = x.astype(_to_mlx_dtype(dtype))
    # MLX raises a raw IndexError for an out-of-bounds axis; the op contract
    # requires a ValueError, so validate axis bounds up front.
    axes = axis if isinstance(axis, tuple) else (axis,)
    for ax in axes:
        if ax is not None and not (-x.ndim <= ax < x.ndim):
            raise ValueError(
                f"Invalid axis {ax} for array with {x.ndim} dimensions."
            )
    # MLX linalg ops are CPU-only; the spectral/nuclear matrix norms compute an
    # internal SVD that aborts on the GPU stream, so force a CPU stream.
    out = mx.linalg.norm(
        x, ord=ord, axis=axis, keepdims=keepdims, stream=mx.cpu
    )
    # For ord=None, axis=None, keepdims=True on a >=2D input, MLX flattens to a
    # vector first and only restores one dimension. NumPy keeps every reduced
    # dimension, so restore the full shape.
    if ord is None and axis is None and keepdims and x.ndim >= 2:
        out = mx.reshape(out, (1,) * x.ndim)
    return out


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    x = convert_to_tensor(x)
    m, n = x.shape[-2], x.shape[-1]
    # MLX linalg ops are CPU-only; force a CPU compute stream.
    if mode == "reduced" or n >= m:
        # MLX only has reduced QR; for n >= m it already equals complete mode.
        return mx.linalg.qr(x, stream=mx.cpu)
    # Tall matrix complete mode: MLX cannot produce the (m, m) orthonormal Q
    # natively. Emulate it by factoring [A | I] and keeping the first m
    # columns of Q (an orthonormal basis of R^m) and the first n columns of R.
    eye = mx.broadcast_to(
        mx.eye(m, dtype=x.dtype), tuple(x.shape[:-2]) + (m, m)
    )
    augmented = mx.concatenate([x, eye], axis=-1)
    q, r = mx.linalg.qr(augmented, stream=mx.cpu)
    return q[..., :, :m], r[..., :, :n]


def solve(a, b):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    # MLX linalg ops are CPU-only; force a CPU compute stream.
    return mx.linalg.solve(a, b, stream=mx.cpu)


def solve_triangular(a, b, lower=False):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    # MLX's signature uses `upper`, not `lower`; and the op is CPU-only.
    if b.ndim == a.ndim - 1:
        b = mx.expand_dims(b, axis=-1)
        return mx.linalg.solve_triangular(
            a, b, upper=not lower, stream=mx.cpu
        ).squeeze(axis=-1)
    return mx.linalg.solve_triangular(a, b, upper=not lower, stream=mx.cpu)


def svd(x, full_matrices=True, compute_uv=True):
    x = convert_to_tensor(x)
    # MLX svd has no `full_matrices` kwarg (it always returns full matrices,
    # matching numpy's full_matrices=True) and is CPU-only.
    if not compute_uv:
        return mx.linalg.svd(x, compute_uv=False, stream=mx.cpu)
    return mx.linalg.svd(x, stream=mx.cpu)


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    # MLX does not have lstsq natively. Use the pseudoinverse approach via the
    # file's own `pinv`, which runs on the CPU stream and honors `rcond`.
    # x = pinv(a) @ b
    a_pinv = pinv(a, rcond=rcond)
    return mx.matmul(a_pinv, b, stream=mx.cpu)


def matrix_rank(x, tol=None):
    x = convert_to_tensor(x)
    if x.ndim < 2:
        raise ValueError(
            "Expected input to have rank >= 2. "
            f"Received input with shape {x.shape}."
        )
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        x = x.astype(_to_mlx_dtype(dtypes.result_type(dtype, "float32")))
    # MLX linalg ops are CPU-only; force a CPU compute stream.
    _, s, _ = mx.linalg.svd(x, stream=mx.cpu)
    if tol is None:
        eps = mx.finfo(s.dtype).eps
        tol = mx.max(s, axis=-1, keepdims=True) * max(x.shape[-2:]) * eps
    return mx.sum(s > tol, axis=-1).astype(mx.int32)


def pinv(x, rcond=None):
    x = convert_to_tensor(x)
    if x.ndim < 2:
        raise ValueError(
            "Expected input to have rank >= 2. "
            f"Received input with shape {x.shape}."
        )
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        x = x.astype(_to_mlx_dtype(dtypes.result_type(dtype, "float32")))
    # MLX linalg ops are CPU-only; force a CPU compute stream.
    if rcond is None:
        return mx.linalg.pinv(x, stream=mx.cpu)
    # mx.linalg.pinv has no rcond/rtol; apply the SVD-threshold cutoff
    # manually: zero out singular values <= rcond * max(singular value).
    u, s, vt = mx.linalg.svd(x, stream=mx.cpu)
    k = s.shape[-1]
    u = u[..., :, :k]
    vt = vt[..., :k, :]
    cutoff = rcond * mx.max(s, axis=-1, keepdims=True)
    s_inv = mx.where(s > cutoff, mx.reciprocal(s), mx.zeros_like(s))
    scaled = mx.swapaxes(vt, -1, -2) * s_inv[..., None, :]
    # Run the reconstruction matmul on the CPU stream: the GPU float32 matmul
    # is too inaccurate for the test atol (same Metal matmul precision issue).
    return mx.matmul(scaled, mx.swapaxes(u, -1, -2), stream=mx.cpu)


def jvp(fun, primals, tangents, has_aux=False):
    primals = list(primals)
    tangents = list(tangents)
    if not has_aux:
        outs, jvps = mx.jvp(fun, primals, tangents)
        return outs[0], jvps[0]
    # `mx.jvp` differentiates every output of `fun`; for has_aux the second
    # return value must be carried through untouched, so differentiate only the
    # primal output and compute the aux separately on the primals.
    outs, jvps = mx.jvp(lambda *p: fun(*p)[0], primals, tangents)
    _, aux = fun(*primals)
    return outs[0], jvps[0], aux
