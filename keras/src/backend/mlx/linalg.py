import mlx.core as mx

from keras.src.backend import standardize_dtype
from keras.src.backend.common import dtypes
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import _to_mlx_dtype


def cholesky(a, upper=False):
    out = mx.linalg.cholesky(a, upper=upper)
    return out


def cholesky_inverse(a, upper=False):
    identity = mx.eye(a.shape[-1], dtype=a.dtype)
    inv_chol = solve_triangular(a, identity, lower=not upper)
    if upper:
        a_inv = mx.matmul(inv_chol, mx.transpose(inv_chol))
    else:
        a_inv = mx.matmul(mx.transpose(inv_chol), inv_chol)
    return a_inv


def det(a):
    # MLX does not have a native det. Compute via LU decomposition:
    # P @ L @ U = A, so det(A) = det(P) * det(L) * det(U).
    # det(L) = 1 (unit lower triangular), det(U) = prod(diag(U)),
    # det(P) = sign of the permutation encoded by P.
    p, l, u = mx.linalg.lu(a)
    diag_u = mx.diagonal(u, axis1=-2, axis2=-1)
    det_u = mx.prod(diag_u, axis=-1)

    n = a.shape[-1]
    # Extract permutation indices from the permutation matrix P.
    p_indices = mx.argmax(p, axis=-1)

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
    return mx.linalg.eig(a)


def eigh(a):
    return mx.linalg.eigh(a)


def inv(a):
    return mx.linalg.inv(a)


def lu_factor(a):
    return mx.linalg.lu_factor(a)


def norm(x, ord=None, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    dtype = standardize_dtype(x.dtype)
    if "int" in dtype or dtype == "bool":
        dtype = dtypes.result_type(x.dtype, "float32")
        x = x.astype(_to_mlx_dtype(dtype))
    return mx.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def qr(x, mode="reduced"):
    if mode not in {"reduced", "complete"}:
        raise ValueError(
            "`mode` argument value not supported. "
            "Expected one of {'reduced', 'complete'}. "
            f"Received: mode={mode}"
        )
    return mx.linalg.qr(x, mode=mode)


def solve(a, b):
    return mx.linalg.solve(a, b)


def solve_triangular(a, b, lower=False):
    if b.ndim == a.ndim - 1:
        b = mx.expand_dims(b, axis=-1)
        return mx.linalg.solve_triangular(a, b, lower=lower).squeeze(axis=-1)
    return mx.linalg.solve_triangular(a, b, lower=lower)


def svd(x, full_matrices=True, compute_uv=True):
    if not compute_uv:
        # MLX svd always returns (U, S, Vt), extract just S
        _, s, _ = mx.linalg.svd(x, full_matrices=False)
        return s
    return mx.linalg.svd(x, full_matrices=full_matrices)


def lstsq(a, b, rcond=None):
    a = convert_to_tensor(a)
    b = convert_to_tensor(b)
    # MLX does not have lstsq natively. Use pseudoinverse approach.
    # x = pinv(a) @ b
    a_pinv = mx.linalg.pinv(a)
    return mx.matmul(a_pinv, b)


def jvp(fun, primals, tangents, has_aux=False):
    raise NotImplementedError("JVP is not supported by the MLX backend.")
