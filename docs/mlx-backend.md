# MLX Backend for Keras 3

## Overview

This document describes the MLX backend implementation for Keras 3, enabling inference and model execution on Apple Silicon via `KERAS_BACKEND=mlx`.

[MLX](https://github.com/ml-explore/mlx) is Apple's array framework for machine learning on Apple Silicon. It shares a similar design philosophy with JAX — lazy evaluation, functional transforms, and a NumPy-like API — making it a natural fit as a Keras backend.

## Quick Start

```bash
pip install mlx
export KERAS_BACKEND=mlx

python -c "
import keras
import numpy as np

model = keras.Sequential([
    keras.Input(shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

x = np.random.randn(32, 784).astype('float32')
predictions = model.predict(x, verbose=0)
print(predictions.shape)  # (32, 10)
"
```

## Architecture

The backend follows the same structure as existing Keras backends (TensorFlow, JAX, PyTorch, NumPy, OpenVINO):

```
keras/src/backend/mlx/
    __init__.py       # Public API surface, re-exports from submodules
    core.py           # Variable, convert_to_tensor, scan, scatter, dtype handling
    numpy.py          # 215 numpy-compatible array operations
    nn.py             # Neural network ops: activations, conv, pooling, losses, attention
    math.py           # FFT, segment ops, top_k, STFT/ISTFT
    linalg.py         # Linear algebra: cholesky, SVD, QR, solve, det
    image.py          # Image ops: resize, affine_transform, gaussian_blur
    random.py         # Random distributions: normal, uniform, dropout, etc.
    rnn.py            # RNN loop, LSTM, GRU cell implementations
    trainer.py        # MLXTrainer: fit(), evaluate(), predict() with gradient-based training
    layer.py          # MLXLayer mixin (empty, no special behavior needed)
    export.py         # MLXExportArchive (stubs, export not supported)
```

### Registration Points

The backend is registered in the following files:

| File | Purpose |
|------|---------|
| `keras/src/backend/__init__.py` | Backend module import |
| `keras/src/utils/backend_utils.py` | Validation, DynamicBackend support |
| `keras/src/models/model.py` | MLXTrainer binding |
| `keras/src/layers/layer.py` | MLXLayer binding |
| `keras/src/export/saved_model.py` | MLXExportArchive binding |
| `conftest.py` | Non-trainable marking, exclusion list |
| `api_gen.py` | Excluded from API codegen |
| `integration_tests/import_test.py` | Import smoke test |
| `integration_tests/numerical_test.py` | Skipped (requires training) |
| `requirements.txt` | `mlx;sys_platform == 'darwin'` |

## Implementation Details

### Design Principles

1. **MLX-first**: All computation uses `mlx.core`. NumPy is used only as an I/O bridge (`convert_to_tensor` input, `convert_to_numpy` output) and for a handful of operations MLX doesn't support natively.

2. **No float64**: MLX doesn't support `float64`. All `float64` requests are silently downcast to `float32` with a one-time warning. This is handled centrally in `_to_mlx_dtype()`.

3. **No `mx.flip`**: MLX doesn't have a `flip` function. A `_flip()` helper in `core.py` implements it via slicing (`x[::-1]`).

4. **Lazy evaluation**: MLX uses lazy evaluation like JAX. `mx.eval()` is called explicitly when values need to be materialized (e.g., before converting to NumPy, or for data-dependent shapes).

### Module Breakdown

#### `core.py` (488 lines)

The foundation module. Key components:

- **`Variable`**: Wraps `KerasVariable` with MLX array storage
- **`convert_to_tensor`**: Converts Python/NumPy values to `mx.array`, handling dtype promotion
- **`_to_mlx_dtype`**: Maps Keras dtype strings to MLX dtype objects, with float64 downcast
- **`_flip`**: Replacement for missing `mx.flip`, uses slicing
- **`compute_output_spec`**: Symbolic shape inference using probe tensors
- **`scan`**, **`associative_scan`**: Functional scan operations
- **`scatter`**, **`scatter_update`**: Scatter operations using `mx.array.at[].add()`
- **`slice`**, **`slice_update`**: Tensor slicing with Python `builtins.slice`
- **`custom_gradient`**: Wraps Keras gradient convention into MLX's `mx.custom_function`

#### `numpy.py` (2260 lines)

215 NumPy-compatible functions. Most map directly to `mlx.core` equivalents:

```python
# Direct mapping (majority of functions)
def zeros(shape, dtype=None):
    return mx.zeros(shape, dtype=_to_mlx_dtype(dtype or config.floatx()))

# Two-input ops follow dtype promotion
def add(x1, x2):
    dtype = dtypes.result_type(...)
    return mx.add(convert_to_tensor(x1, dtype), convert_to_tensor(x2, dtype))
```

Functions without direct MLX equivalents use numpy bridge:
- `nonzero`, `unique` (data-dependent output shapes)
- `searchsorted`, `digitize` (binary search)
- `histogram`, `bincount`, `correlate`
- `gcd`, `lcm`, `nextafter`, `i0`

NaN-aware operations (`nansum`, `nanmax`, etc.) use `mx.where(mx.isnan(x), fill, x)` pattern.

#### `nn.py` (1951 lines)

Neural network operations:

- **Activations** (27): All standard activations (`relu`, `gelu`, `silu`, `softmax`, etc.) implemented with MLX primitives
- **Convolution**: Uses `mx.conv_general` with kernel transposition from Keras format `(*spatial, in_ch, out_ch)` to MLX format `(out_ch, *spatial, in_ch)`
- **Pooling**: Sliding-window gather using `mx.arange` index arrays
- **Losses**: `categorical_crossentropy`, `binary_crossentropy`, `sparse_categorical_crossentropy`, `ctc_loss`
- **Attention**: `dot_product_attention` with einsum-based implementation
- **Normalization**: `batch_normalization`, `moments`

#### `random.py` (115 lines)

All distributions use `mlx.core.random`:
- `normal`, `uniform`, `truncated_normal`, `randint`, `categorical`, `dropout`, `shuffle`
- `gamma`, `binomial`, `beta` use numpy bridge (MLX lacks native implementations)

#### `rnn.py` (408 lines)

Sequential scan implementations:
- **`rnn`**: Generic RNN loop with mask support
- **`lstm`**: Optimized with pre-computed input projections
- **`gru`**: Supports both `reset_after=True` and `reset_after=False`

#### `math.py` (378 lines)

- **FFT**: `fft`, `fft2`, `ifft2`, `rfft`, `irfft` via `mlx.core.fft`
- **STFT/ISTFT**: Windowed FFT with overlap-add
- **Segment ops**: `segment_sum`, `segment_max` via scatter
- **`top_k`**: Via `mx.argsort` + reversal

#### `linalg.py` (135 lines)

- Direct MLX mappings: `cholesky`, `eig`, `eigh`, `inv`, `qr`, `solve`, `svd`, `norm`
- Custom implementations: `det` (via LU decomposition), `lstsq` (via pseudoinverse)

#### `image.py` (1221 lines)

All implemented from MLX primitives or with minimal numpy bridge:
- **Color**: `rgb_to_grayscale`, `rgb_to_hsv`, `hsv_to_rgb`
- **Spatial**: `resize`, `affine_transform`, `map_coordinates`
- **Filters**: `gaussian_blur` (via `mx.conv2d`)

#### `trainer.py` (~500 lines)

Full training support using `mx.value_and_grad` for gradient computation:
- `fit()`: Full training loop with callbacks, validation, epoch iteration
- `train_step()`: Forward pass + gradient computation via `mx.value_and_grad`, optimizer apply
- `evaluate()`: Test step with metric accumulation
- `predict()`: Batch iteration with MLX tensor concatenation
- `train_on_batch()`, `test_on_batch()`, `predict_on_batch()`: Single-batch variants

## Test Results

Test suite run with `KERAS_BACKEND=mlx pytest keras/src/ops/`:

| Test Suite | Passed | Failed | Skipped | Pass Rate |
|-----------|--------|--------|---------|-----------|
| Core ops | 155 | 0 | 14 | 100% |
| NumPy ops | ~5006 | ~261 | ~700 | ~95% |
| NN ops | 307 | 41 | 16 | ~88% |
| Math ops | ~200 | ~50 | — | ~80% |
| Linalg ops | ~268 | ~94 | — | ~74% |
| Image ops | varies | varies | — | — |
| **Total** | **~6000+** | **~476** | **~744** | **~83%** |

Known failure categories:
- **float8 dtypes**: MLX doesn't support `float8_e4m3fn` / `float8_e5m2`
- **Precision**: Some numerical tests fail due to float64 -> float32 downcast
- **Missing MLX features**: Some edge cases in linalg, image transforms

An exclusion list is maintained at `keras/src/backend/mlx/excluded_concrete_tests.txt`.

## What's Not Yet Implemented

### Export

Model export (`SavedModel`, ONNX) is not supported. The `MLXExportArchive` raises `NotImplementedError`.

### Distribution

Distributed training / multi-device is not supported (`distribution_lib = None`).

### Remaining Test Failures

The ~476 test failures fall into these categories:

| Category | Count (approx) | Root Cause |
|----------|-------|------------|
| float8 dtypes | ~5 | MLX has no float8 support |
| Precision (float64) | ~50 | float64 downcast to float32 |
| Image ops | ~30 | Affine/perspective transform edge cases |
| Linalg edge cases | ~90 | QR complete mode, SVD arg handling |
| Conv/pool edge cases | ~20 | Transposed conv padding, asymmetric cases |
| Dtype promotion | ~100+ | Some dtype combinations differ from NumPy |
| CTC loss | ~5 | Implementation differences |
| Misc | ~150 | Various small incompatibilities |

### Functions Using NumPy Bridge

These functions convert to NumPy internally for computation that MLX doesn't support natively. They still return MLX tensors:

- `random.gamma`, `random.binomial`, `random.beta` — MLX lacks these distributions
- `numpy.nonzero`, `numpy.unique` — data-dependent output shapes
- `numpy.searchsorted`, `numpy.digitize`, `numpy.histogram`
- `numpy.gcd`, `numpy.lcm`, `numpy.correlate`, `numpy.corrcoef`
- `numpy.pad` (reflect/symmetric modes only)
- `numpy.repeat` (array repeats only; int repeats use MLX)
- `numpy.moveaxis` (tuple source/dest only; int uses MLX)
- `image.map_coordinates` — scipy bridge
- `math.stft` — reflect padding via numpy
- `math.qr(mode="complete")` — MLX only supports reduced QR
- `linalg.det` — permutation sign computation

## Platform Requirements

- **OS**: macOS (Darwin)
- **Hardware**: Apple Silicon (M1/M2/M3/M4) recommended; Intel Macs may work with reduced performance
- **Python**: 3.11+
- **MLX**: 0.31.1+
