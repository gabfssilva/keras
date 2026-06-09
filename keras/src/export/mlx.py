from keras.src import backend
from keras.src import tree
from keras.src.export.export_utils import convert_spec_to_tensor
from keras.src.export.export_utils import get_input_signature
from keras.src.utils import io_utils


def export_mlx(
    model,
    filepath,
    verbose=None,
    input_signature=None,
    **kwargs,
):
    """Export the model as a native MLX artifact for inference.

    This method exports the model's forward pass (its `call()` method) as a
    standalone MLX function file (`.mlxfn`), serialized with
    `mlx.core.export_function`. The artifact bundles the computation graph
    together with the model weights and can be reloaded with
    `mlx.core.import_function` from any MLX front-end (Python, C++, Swift),
    without the original model code.

    Args:
        filepath: `str` or `pathlib.Path` object. The path to save the artifact.
        verbose: `bool`. Whether to print a message during export. Defaults to
            `None`, which is treated as `True`.
        input_signature: Optional. Specifies the shape and dtype of the model
            inputs. Can be a structure of `keras.InputSpec`,
            `backend.KerasTensor`, or backend tensor. If not provided, it will
            be automatically computed. Defaults to `None`.
        **kwargs: Additional keyword arguments:
            - `shapeless`: `bool`. If `True`, export a function that accepts
                inputs with variable shapes (e.g. a dynamic batch size); the
                `input_signature` is then only used to trace the graph. If
                `False`, the function is specialized to the traced shapes. If
                omitted, it is inferred from `input_signature`: shapeless when
                any dimension is dynamic (e.g. a `None` batch size), and
                specialized otherwise.

    **Note:** This feature is only supported with the MLX backend. The exported
    artifact requires an MLX runtime on the target platform; it is not a
    TensorFlow SavedModel or ONNX graph.

    Example:

    ```python
    # Export the model as a native MLX artifact
    model.export("path/to/model.mlxfn", format="mlx")

    # Load the artifact in a different process/environment
    import mlx.core as mx
    fn = mx.import_function("path/to/model.mlxfn")
    outputs = fn(mx.array(input_data))
    ```
    """
    import mlx.core as mx

    if backend.backend() != "mlx":
        raise NotImplementedError(
            "`export_mlx` is only compatible with the MLX backend. "
            f"Received: backend={backend.backend()}"
        )

    actual_verbose = True if verbose is None else verbose
    shapeless = kwargs.pop("shapeless", None)
    if kwargs:
        raise ValueError(f"Unsupported keyword arguments: {list(kwargs)}")

    if input_signature is None:
        input_signature = get_input_signature(model)

    flat_specs = tree.flatten(input_signature)

    # When the inputs have dynamic dimensions (e.g. a `None` batch size),
    # export a shapeless function so the artifact accepts variable shapes. A
    # fully-static `input_signature` produces a shape-specialized function.
    if shapeless is None:
        shapeless = any(
            dim is None for spec in flat_specs for dim in spec.shape
        )

    # Trace with concrete sample inputs; dynamic dims are replaced with 1.
    sample_inputs = [
        convert_spec_to_tensor(spec, replace_none_number=1)
        for spec in flat_specs
    ]

    # Materialize the weights so the exported graph captures their values
    # rather than their initialization computation.
    mx.eval(*(variable.value for variable in model.variables))

    # The exported function receives the inputs as flat positional args and
    # repacks them into the structure the model expects before calling it.
    def serving_fn(*flat_args):
        inputs = tree.pack_sequence_as(input_signature, list(flat_args))
        if len(inputs) == 1:
            return model(inputs[0], training=False)
        return model(*inputs, training=False)

    mx.export_function(
        str(filepath), serving_fn, *sample_inputs, shapeless=shapeless
    )

    if actual_verbose:
        io_utils.print_msg(f"Saved artifact at '{filepath}'.")
