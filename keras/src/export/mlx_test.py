"""Tests for native MLX exporting utilities."""

import os

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src import tree


def get_model(model_type="sequential"):
    if model_type == "sequential":
        return models.Sequential(
            [
                layers.Input((4,)),
                layers.Dense(8, activation="relu"),
                layers.Dense(3),
            ]
        )
    elif model_type == "functional":
        x = layers.Input((4,))
        y = layers.Dense(8, activation="relu")(x)
        y = layers.Dense(3)(y)
        return models.Model(x, y)
    else:
        raise ValueError(f"Unknown model_type={model_type}")


@pytest.mark.skipif(
    backend.backend() != "mlx",
    reason="The MLX export format requires the MLX backend.",
)
class ExportMLXTest(testing.TestCase):
    def _reimport_and_run(self, filepath, inputs):
        import mlx.core as mx

        fn = mx.import_function(filepath)
        args = [
            mx.array(np.asarray(x, dtype="float32"))
            for x in tree.flatten(inputs)
        ]
        out = fn(*args)
        out = out[0] if isinstance(out, (list, tuple)) else out
        return np.array(out)

    @parameterized.named_parameters(
        ("sequential", "sequential"),
        ("functional", "functional"),
    )
    def test_standard_model_export(self, model_type):
        model = get_model(model_type)
        ref_input = np.random.rand(3, 4).astype("float32")
        ref_output = ops.convert_to_numpy(model(ref_input))

        temp_filepath = os.path.join(self.get_temp_dir(), "model.mlxfn")
        model.export(temp_filepath, format="mlx")

        out = self._reimport_and_run(temp_filepath, ref_input)
        self.assertAllClose(ref_output, out, atol=1e-5)

    def test_model_with_multiple_inputs(self):
        a = layers.Input((4,), name="a")
        b = layers.Input((4,), name="b")
        out = layers.Dense(3)(layers.Concatenate()([a, b]))
        model = models.Model([a, b], out)

        ref_inputs = [
            np.random.rand(2, 4).astype("float32"),
            np.random.rand(2, 4).astype("float32"),
        ]
        ref_output = ops.convert_to_numpy(model(ref_inputs))

        temp_filepath = os.path.join(self.get_temp_dir(), "model.mlxfn")
        model.export(temp_filepath, format="mlx")

        out = self._reimport_and_run(temp_filepath, ref_inputs)
        self.assertAllClose(ref_output, out, atol=1e-5)

    def test_dynamic_batch_is_shapeless(self):
        # A dynamic batch size (`None`) should export a shapeless function that
        # accepts a batch size different from the one used to trace.
        model = get_model("sequential")
        temp_filepath = os.path.join(self.get_temp_dir(), "model.mlxfn")
        model.export(temp_filepath, format="mlx")

        big_input = np.random.rand(7, 4).astype("float32")
        ref_output = ops.convert_to_numpy(model(big_input))
        out = self._reimport_and_run(temp_filepath, big_input)
        self.assertEqual(out.shape, (7, 3))
        self.assertAllClose(ref_output, out, atol=1e-5)

    def test_explicit_shapeless_false_is_specialized(self):
        model = get_model("sequential")
        temp_filepath = os.path.join(self.get_temp_dir(), "model.mlxfn")
        model.export(temp_filepath, format="mlx", shapeless=False)

        ref_input = np.random.rand(1, 4).astype("float32")
        ref_output = ops.convert_to_numpy(model(ref_input))
        out = self._reimport_and_run(temp_filepath, ref_input)
        self.assertAllClose(ref_output, out, atol=1e-5)


@pytest.mark.skipif(
    backend.backend() == "mlx",
    reason="Tests the error raised on non-MLX backends.",
)
class ExportMLXWrongBackendTest(testing.TestCase):
    def test_export_mlx_requires_mlx_backend(self):
        model = get_model("sequential")
        temp_filepath = os.path.join(self.get_temp_dir(), "model.mlxfn")
        with self.assertRaisesRegex(
            ValueError, "MLX export requires the MLX backend."
        ):
            model.export(temp_filepath, format="mlx")
