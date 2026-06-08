import mlx.core as mx
import mlx.core.random as mlx_random

from keras.src.backend.config import floatx
from keras.src.backend.mlx.core import _to_mlx_dtype
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.random.seed_generator import SeedGenerator  # noqa: F401
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed  # noqa: F401


def _get_key(seed):
    seed = draw_seed(seed)
    return mx.array(seed, dtype=mx.uint32)


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    key = _get_key(seed)
    sample = mlx_random.normal(shape=shape, key=key)
    sample = sample * stddev + mean
    return sample.astype(_to_mlx_dtype(dtype))


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    key = _get_key(seed)
    sample = mlx_random.uniform(
        shape=shape, low=minval, high=maxval, key=key
    )
    return sample.astype(_to_mlx_dtype(dtype))


def categorical(logits, num_samples, dtype="int64", seed=None):
    key = _get_key(seed)
    logits = convert_to_tensor(logits)
    output = mlx_random.categorical(
        logits, num_samples=num_samples, key=key
    )
    return output.astype(_to_mlx_dtype(dtype))


def randint(shape, minval, maxval, dtype="int32", seed=None):
    key = _get_key(seed)
    sample = mlx_random.randint(
        low=minval, high=maxval, shape=shape, key=key
    )
    return sample.astype(_to_mlx_dtype(dtype))


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    key = _get_key(seed)
    sample = mlx_random.truncated_normal(
        lower=-2.0, upper=2.0, shape=shape, key=key
    )
    sample = sample * stddev + mean
    return sample.astype(_to_mlx_dtype(dtype))


def dropout(inputs, rate, noise_shape=None, seed=None):
    if rate == 1.0:
        return mx.zeros_like(inputs)
    if rate == 0.0:
        return inputs
    key = _get_key(seed)
    keep_prob = 1.0 - rate
    if noise_shape is None:
        noise_shape = inputs.shape
    else:
        noise_shape = [
            n if n is not None else inputs.shape[i]
            for i, n in enumerate(noise_shape)
        ]
    mask = mlx_random.uniform(shape=noise_shape, key=key) < keep_prob
    mask = mx.broadcast_to(mask, inputs.shape)
    return mx.where(mask, inputs / keep_prob, mx.zeros_like(inputs))


def shuffle(x, axis=0, seed=None):
    key = _get_key(seed)
    x = convert_to_tensor(x)
    return mlx_random.permutation(x, axis=axis, key=key)


def gamma(shape, alpha, dtype=None, seed=None):
    dtype = dtype or floatx()
    key = _get_key(seed)
    alpha = convert_to_tensor(alpha, dtype="float32")
    alpha = mx.broadcast_to(alpha, shape)
    # Marsaglia-Tsang method for gamma(alpha >= 1)
    # For alpha < 1: gamma(alpha) = gamma(alpha+1) * U^(1/alpha)
    squeeze_mask = alpha < 1.0
    alpha_adj = mx.where(squeeze_mask, alpha + 1.0, alpha)
    d = alpha_adj - 1.0 / 3.0
    c = 1.0 / mx.sqrt(9.0 * d)
    result = mx.zeros(shape, dtype=mx.float32)
    # Rejection sampling — iterate enough times for convergence
    remaining = mx.ones(shape, dtype=mx.bool_)
    for _ in range(100):
        keys = mx.random.split(key)
        k1, key = keys[0], keys[1]
        keys2 = mx.random.split(key)
        k2, key = keys2[0], keys2[1]
        z = mx.random.normal(shape=shape, key=k1)
        u = mx.random.uniform(shape=shape, key=k2)
        v = (1.0 + c * z) ** 3
        valid = z > -1.0 / c
        accept = mx.log(u) < (
            0.5 * z * z + d - d * v + d * mx.log(mx.maximum(v, 1e-30))
        )
        accept = mx.logical_and(accept, valid)
        accept = mx.logical_and(accept, remaining)
        result = mx.where(accept, d * v, result)
        remaining = mx.logical_and(remaining, mx.logical_not(accept))
        mx.eval(remaining)
        if not mx.any(remaining):
            break
    # Adjust for alpha < 1
    k3 = key
    u_adj = mx.random.uniform(shape=shape, key=k3)
    result = mx.where(
        squeeze_mask,
        result * mx.power(u_adj, 1.0 / mx.where(squeeze_mask, alpha, 1.0)),
        result,
    )
    return result.astype(_to_mlx_dtype(dtype))


def binomial(shape, counts, probabilities, dtype=None, seed=None):
    dtype = dtype or floatx()
    key = _get_key(seed)
    counts = convert_to_tensor(counts, dtype="float32")
    probabilities = convert_to_tensor(probabilities, dtype="float32")
    counts = mx.broadcast_to(counts, shape)
    probabilities = mx.broadcast_to(probabilities, shape)
    # Sum of Bernoulli trials
    mx.eval(counts)
    max_n = int(mx.max(counts))
    if max_n == 0:
        return mx.zeros(shape, dtype=_to_mlx_dtype(dtype))
    # Generate max_n Bernoulli trials and sum
    trials = mx.random.uniform(
        shape=(max_n,) + tuple(shape), key=key
    )
    successes = (trials < mx.expand_dims(probabilities, 0)).astype(mx.float32)
    # Mask out trials beyond each element's count
    trial_idx = mx.arange(max_n).reshape((-1,) + (1,) * len(shape))
    mask = trial_idx < mx.expand_dims(counts, 0)
    successes = successes * mask.astype(mx.float32)
    result = mx.sum(successes, axis=0)
    return result.astype(_to_mlx_dtype(dtype))


def beta(shape, alpha, beta, dtype=None, seed=None):
    dtype = dtype or floatx()
    key = _get_key(seed)
    k1, k2 = mx.random.split(key)
    # beta(a, b) = g1 / (g1 + g2) where g1 ~ gamma(a), g2 ~ gamma(b)
    g1 = gamma(shape, alpha, dtype="float32", seed=int(k1[0]))
    g2 = gamma(shape, beta, dtype="float32", seed=int(k2[0]))
    result = g1 / (g1 + g2)
    return result.astype(_to_mlx_dtype(dtype))
