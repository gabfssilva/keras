"""Tests for the MLX backend distribution_lib (single-process hooks).

Multi-process data-parallel correctness (gradient all-reduce keeping replicas
in sync) is validated separately by `integration_tests/mlx_data_parallel.py`,
which must be launched with `mlx.launch` so each rank is its own process.
"""

import pytest

from keras.src import backend
from keras.src import distribution as keras_distribution
from keras.src import testing


@pytest.mark.skipif(
    backend.backend() != "mlx",
    reason="The MLX distribution_lib is only available on the MLX backend.",
)
class MLXDistributionLibTest(testing.TestCase):
    def test_single_process_world(self):
        from keras.src.backend import distribution_lib

        self.assertEqual(distribution_lib.num_processes(), 1)
        self.assertEqual(distribution_lib.process_id(), 0)
        self.assertEqual(distribution_lib.get_device_count(), 1)
        self.assertEqual(len(distribution_lib.list_devices()), 1)

    def test_distribute_tensor_is_replicated_noop(self):
        import mlx.core as mx

        from keras.src.backend import distribution_lib

        x = mx.array([1.0, 2.0, 3.0])
        out = distribution_lib.distribute_tensor(x, layout=None)
        self.assertTrue(bool(mx.all(out == x).item()))

    def test_distribute_data_input_is_identity_for_one_process(self):
        import mlx.core as mx

        from keras.src.backend import distribution_lib

        batch = mx.zeros((8, 4))
        out = distribution_lib.distribute_data_input(
            batch, layout=None, batch_dim_name="batch"
        )
        self.assertEqual(out.shape, (8, 4))

    def test_backend_mesh_and_layout_roundtrip(self):
        mesh = keras_distribution.DeviceMesh(
            shape=(1,),
            axis_names=("batch",),
            devices=keras_distribution.list_devices(),
        )
        self.assertIsNotNone(mesh.backend_mesh)
        layout = keras_distribution.TensorLayout(
            ("batch", None), device_mesh=mesh
        )
        self.assertEqual(tuple(layout.backend_layout), ("batch", None))

    def test_data_parallel_constructs(self):
        mesh = keras_distribution.DeviceMesh(
            shape=(1,),
            axis_names=("batch",),
            devices=keras_distribution.list_devices(),
        )
        dp = keras_distribution.DataParallel(device_mesh=mesh)
        self.assertIsNotNone(dp)
