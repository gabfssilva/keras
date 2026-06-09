"""Utilities for distribution strategy with MLX backend."""

import collections

import mlx.core as mx

from keras.src.random import seed_generator
from keras.src.utils import rng_utils

_MLXMesh = collections.namedtuple("_MLXMesh", ["shape", "axis_names"])


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    MLX runs on a single Metal GPU per process, so the cluster is described
    with one logical device per process. The length of the returned list is
    what `DeviceMesh` validation relies on, not the string content.

    Args:
        device_type: Unused for MLX; accepted for API parity.

    Return:
        List of devices that are available for distribute computation.
    """
    count = num_processes()
    return [f"gpu:{i}" for i in range(count)]


def get_device_count(device_type=None):
    """Returns the number of available MLX devices.

    For MLX there is one logical device per process.

    Args:
        device_type: Unused for MLX; accepted for API parity.

    Returns:
        int: The total number of processes in the cluster.
    """
    return num_processes()


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    DataParallel layouts are always fully replicated, and MLX has no
    compiler-inserted sharding, so this is a no-op.

    Args:
        tensor: The tensor to distribute.
        layout: `TensorLayout` for the variable (replicated).

    Returns:
        The tensor unchanged.
    """
    return tensor


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Shard the input batch along the batch dimension by process.

    Each MLX process holds its own local batch; there is no global-array
    abstraction. The gradient all-reduce in the trainer is what couples the
    replicas. For DataParallel the data adapter already produces a per-process
    shard, so this slices the batch along its first axis only when it still
    carries the full global batch.

    Args:
        per_process_batch: The local batch for this process.
        layout: `TensorLayout` for the data (sharded on the batch dim).
        batch_dim_name: Name of the batch dimension.

    Returns:
        The per-process shard of the batch.
    """
    world_size = num_processes()
    if world_size <= 1:
        return per_process_batch

    rank = process_id()
    batch_size = per_process_batch.shape[0]
    if batch_size % world_size != 0:
        raise ValueError(
            "The batch size must be divisible by the number of processes. "
            f"Received: batch_size={batch_size}, "
            f"num_processes={world_size}"
        )
    shard_size = batch_size // world_size
    start = rank * shard_size
    return per_process_batch[start : start + shard_size]


def initialize_rng():
    """Initializes the global random number generator across processes.

    Variable initializers bake a `make_default_seed()` draw from the Python
    global RNG at construction time, so identical initial weights across ranks
    require an identical global seed on every process before any layer is
    built. This sums per-process seeds into a single shared value, mirroring
    the JAX backend's `psum`.
    """
    global_seed = rng_utils.get_random_seed()
    # Only derive a seed if the user has not set one explicitly via
    # keras.utils.set_random_seed().
    if global_seed is None:
        group = mx.distributed.init(strict=False)
        local_seed = mx.array(
            [seed_generator.make_default_seed()], dtype=mx.uint32
        )
        summed = mx.distributed.all_sum(local_seed, group=group)
        mx.eval(summed)
        global_seed = int(summed.item()) & 0xFFFFFFFF
        rng_utils.set_random_seed(global_seed)


def initialize(job_addresses, num_processes, process_id):
    """Initialize the MLX distributed group.

    MLX derives cluster size and rank from its launcher (e.g. `mlx.launch`
    or MPI) via the environment, so `job_addresses`, `num_processes` and
    `process_id` are accepted for API parity but not used to configure the
    group.

    Args:
        job_addresses: Unused for MLX.
        num_processes: Unused for MLX.
        process_id: Unused for MLX.
    """
    mx.distributed.init(strict=False, backend="any")
    initialize_rng()


def num_processes():
    """Return the number of processes for the current distribution setting."""
    return mx.distributed.init(strict=False).size()


def process_id():
    """Return the current process ID for the distribution setting."""
    return mx.distributed.init(strict=False).rank()


def _to_backend_device(device_name):
    return device_name


def _to_backend_mesh(device_mesh):
    """Convert the `DeviceMesh` to an MLX backend placeholder.

    MLX has no native mesh type; DataParallel replicates weights and there is
    nothing to map to physical Metal devices. The placeholder only carries the
    shape and axis names for `_to_backend_layout`.

    Args:
        device_mesh: `DeviceMesh` instance to convert.

    Returns:
        A lightweight namedtuple carrying the mesh shape and axis names.
    """
    return _MLXMesh(device_mesh.devices.shape, tuple(device_mesh.axis_names))


def _to_backend_layout(tensor_layout):
    """Convert the `TensorLayout` to an MLX backend placeholder.

    For DataParallel every layout is replicated, so the backend layout is
    simply the tuple of axis names.

    Args:
        tensor_layout: `TensorLayout` instance to convert.

    Returns:
        The tuple of axes.
    """
    return tuple(tensor_layout.axes)
