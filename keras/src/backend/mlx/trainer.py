import warnings

import mlx.core as mx

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src import tree
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.config import max_epochs
from keras.src.backend.mlx import distribution_lib as mlx_distribution_lib
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.backend.mlx.core import is_tensor
from keras.src.distribution import distribution_lib
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.tf_dataset_adapter import TFDatasetAdapter
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils
from keras.src.utils.python_utils import pythonify_logs


class MLXTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def _get_trainable_values(self):
        return [v.value for v in self.trainable_variables]

    def _set_trainable_values(self, values):
        for v, val in zip(self.trainable_variables, values):
            v._value = val

    def _train_state_variables(self):
        return (
            self.trainable_variables,
            self.non_trainable_variables,
            self.optimizer.variables,
            self.metrics_variables,
        )

    def _test_state_variables(self):
        return (
            self.trainable_variables,
            self.non_trainable_variables,
            self.metrics_variables,
        )

    def _predict_state_variables(self):
        return (
            self.trainable_variables,
            self.non_trainable_variables,
        )

    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        def loss_fn(*trainable_values):
            self._set_trainable_values(trainable_values)
            if self._call_has_training_arg:
                y_pred = self(x, training=True)
            else:
                y_pred = self(x)
            loss = self._compute_loss(
                x=x,
                y=y,
                y_pred=y_pred,
                sample_weight=sample_weight,
                training=True,
            )
            return loss, y_pred

        trainable_values = self._get_trainable_values()
        if trainable_values:
            grad_fn = mx.value_and_grad(
                loss_fn, argnums=list(range(len(trainable_values)))
            )
            (loss, y_pred), grads = grad_fn(*trainable_values)
            group = mx.distributed.init(strict=False)
            world_size = group.size()
            if world_size > 1:
                # A single trainable variable yields a bare array (not a
                # tuple), so normalize before averaging across replicas.
                if isinstance(grads, (list, tuple)):
                    grads = [
                        mx.distributed.all_sum(g, group=group) / world_size
                        for g in grads
                    ]
                else:
                    grads = (
                        mx.distributed.all_sum(grads, group=group) / world_size
                    )
            self.optimizer.apply(grads, self.trainable_variables)
        else:
            loss, y_pred = loss_fn()
            warnings.warn("The model does not have any trainable weights.")

        self._loss_tracker.update_state(
            loss,
            sample_weight=next(
                i for i in tree.flatten(x) if i is not None
            ).shape[0],
        )
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        (
            x,
            y,
            sample_weight,
        ) = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        loss = self._compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=False
        )
        self._loss_tracker.update_state(
            loss,
            sample_weight=next(
                i for i in tree.flatten(x) if i is not None
            ).shape[0],
        )
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg:
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        return y_pred

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        def step_function(state, data):
            _restore_state(self._train_state_variables(), state)
            logs = self.train_step(data)
            return logs, _capture_state(self._train_state_variables())

        if not self.run_eagerly and self.jit_compile:
            step_function = mx.compile(step_function)

        def one_step_on_data(data):
            if not self.optimizer.built:
                self.optimizer.build(self.trainable_variables)
            state = _capture_state(self._train_state_variables())
            for single_batch in data:
                logs, state = step_function(state, _convert_batch(single_batch))
                # Force evaluation between batches to avoid Metal command
                # buffer conflicts
                mx.eval(state)
            _restore_state(self._train_state_variables(), state)
            return pythonify_logs(logs)

        self.train_function = one_step_on_data

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        def step_function(state, data):
            _restore_state(self._test_state_variables(), state)
            logs = self.test_step(data)
            return logs, _capture_state(self._test_state_variables())

        if not self.run_eagerly and self.jit_compile:
            step_function = mx.compile(step_function)

        def one_step_on_data(data):
            state = _capture_state(self._test_state_variables())
            for single_batch in data:
                logs, state = step_function(state, _convert_batch(single_batch))
                mx.eval(state)
            _restore_state(self._test_state_variables(), state)
            return pythonify_logs(logs)

        self.test_function = one_step_on_data

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        def step_function(state, data):
            _restore_state(self._predict_state_variables(), state)
            outputs = self.predict_step(data)
            return outputs, _capture_state(self._predict_state_variables())

        if not self.run_eagerly and self.jit_compile:
            step_function = mx.compile(step_function)

        def one_step_on_data(data):
            state = _capture_state(self._predict_state_variables())
            outputs, state = step_function(state, _convert_batch(data[0]))
            for single_batch in data[1:]:
                batch_outputs, state = step_function(
                    state, _convert_batch(single_batch)
                )
                outputs = tree.map_structure(
                    lambda t1, t2: mx.concatenate([t1, t2]),
                    outputs,
                    batch_outputs,
                )
            _restore_state(self._predict_state_variables(), state)
            return outputs

        self.predict_function = one_step_on_data

    def _symbolic_build(self, data_batch=None, iterator=None):
        if data_batch is None and iterator is not None:
            for _, _, data in iterator:
                data_batch = data[0]
                break
            iterator.reset()

        model_unbuilt = not all(layer.built for layer in self._flatten_layers())
        compile_metrics_unbuilt = (
            self._compile_metrics is not None
            and not self._compile_metrics.built
        )
        compile_loss_unbuilt = (
            self._compile_loss is not None and not self._compile_loss.built
        )
        if model_unbuilt or compile_metrics_unbuilt or compile_loss_unbuilt:

            def to_symbolic_input(v):
                if is_tensor(v):
                    return KerasTensor(v.shape, standardize_dtype(v.dtype))
                return v

            data_batch = tree.map_structure(to_symbolic_input, data_batch)
            (
                x,
                y,
                sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(data_batch)
            try:
                y_pred = backend.compute_output_spec(self, x)
            except Exception:
                raise RuntimeError(
                    "Unable to automatically build the model. "
                    "Please build it yourself before calling "
                    "fit/evaluate/predict. "
                    "A model is 'built' when its variables have "
                    "been created and its `self.built` attribute "
                    "is True. Usually, calling the model on a batch "
                    "of data is the right way to build it."
                )
            if compile_metrics_unbuilt:
                backend.compute_output_spec(
                    self.compute_metrics,
                    x,
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                )
            if compile_loss_unbuilt:
                backend.compute_output_spec(
                    self._compute_loss,
                    x,
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                )
        self._post_build()

    @traceback_utils.filter_traceback
    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
        if not self.compiled:
            raise ValueError(
                "You must call `compile()` before calling `fit()`."
            )

        _max_epochs = max_epochs()
        if _max_epochs and _max_epochs < epochs:
            warnings.warn("Limiting epochs to %d" % _max_epochs)
            epochs = _max_epochs

        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            (
                (x, y, sample_weight),
                validation_data,
            ) = array_slicing.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data is not None:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

        epoch_iterator = MLXEpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            steps_per_execution=self.steps_per_execution,
        )

        self._symbolic_build(iterator=epoch_iterator)

        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.stop_training = False
        training_logs = {}
        self.make_train_function()
        callbacks.on_train_begin()
        initial_epoch = self._initial_epoch or initial_epoch
        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)

            logs = {}
            for begin_step, end_step, data in epoch_iterator:
                callbacks.on_train_batch_begin(begin_step)
                logs = self.train_function(data)
                callbacks.on_train_batch_end(end_step, logs)
                if self.stop_training:
                    break

            epoch_logs = dict(self._get_metrics_result_or_logs(logs))

            # Run validation.
            if validation_data is not None and self._should_eval(
                epoch, validation_freq
            ):
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = MLXEpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps_per_execution=self.steps_per_execution,
                        steps_per_epoch=validation_steps,
                        shuffle=False,
                    )
                val_logs = self.evaluate(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    return_dict=True,
                    _use_cached_eval_dataset=True,
                )
                val_logs = {
                    f"val_{name}": val for name, val in val_logs.items()
                }
                epoch_logs.update(val_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break

        if (
            isinstance(self.optimizer, optimizers_module.Optimizer)
            and epochs > 0
        ):
            self.optimizer.finalize_variable_values(self.trainable_weights)

        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.history

    @traceback_utils.filter_traceback
    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            epoch_iterator = MLXEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                steps_per_execution=self.steps_per_execution,
            )

        self._symbolic_build(iterator=epoch_iterator)

        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.make_test_function()
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = {}
        self.reset_metrics()
        for begin_step, end_step, data in epoch_iterator:
            callbacks.on_test_batch_begin(begin_step)
            logs = self.test_function(data)
            callbacks.on_test_batch_end(end_step, logs)
            if self.stop_evaluating:
                break
        logs = pythonify_logs(self._get_metrics_result_or_logs(logs))
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    @traceback_utils.filter_traceback
    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        epoch_iterator = MLXEpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            steps_per_execution=self.steps_per_execution,
        )

        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tree.map_structure(
                    lambda batch_output: [batch_output],
                    batch_outputs,
                )
            else:
                tree.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs,
                    batch_outputs,
                )
            return outputs

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()
        outputs = None
        for begin_step, end_step, data in epoch_iterator:
            callbacks.on_predict_batch_begin(begin_step)
            batch_outputs = self.predict_function(data)
            outputs = append_to_outputs(batch_outputs, outputs)
            callbacks.on_predict_batch_end(end_step, {"outputs": batch_outputs})
            if self.stop_predicting:
                break
        callbacks.on_predict_end()
        return tree.map_structure_up_to(batch_outputs, mx.concatenate, outputs)

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("train_on_batch")
        if class_weight is not None:
            if sample_weight is not None:
                raise ValueError(
                    "Arguments `sample_weight` and `class_weight` "
                    "cannot be specified at the same time. "
                    f"Received: sample_weight={sample_weight}, "
                    f"class_weight={class_weight}"
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )

        data = (x, y, sample_weight)

        self._symbolic_build(data_batch=data)
        self.make_train_function()
        self.reset_metrics()

        logs = self.train_function([data])
        logs = pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        self._assert_compile_called("test_on_batch")

        data = (x, y, sample_weight)

        self._symbolic_build(data_batch=data)
        self.make_test_function()
        self.reset_metrics()

        logs = self.test_function([data])
        logs = pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        self.make_predict_function()
        batch_outputs = self.predict_function([(x,)])
        batch_outputs = tree.map_structure(
            backend.convert_to_numpy, batch_outputs
        )
        return batch_outputs


def _capture_state(variable_groups):
    return [[v.value for v in group] for group in variable_groups]


def _restore_state(variable_groups, state):
    for group, values in zip(variable_groups, state):
        for variable, value in zip(group, values):
            variable._value = value


def _convert_batch(data):
    return tree.map_structure(
        lambda x: x if x is None else convert_to_tensor(x), data
    )


class MLXEpochIterator(EpochIterator):
    def _get_iterator(self):
        distribution = distribution_lib.distribution()
        # `tf.data` inputs are already sharded per process by
        # `distribution.distribute_dataset` inside `TFDatasetAdapter`, and
        # `auto_shard_dataset=False` means the user pre-sharded the data.
        if (
            distribution is not None
            and distribution.auto_shard_dataset
            and mlx_distribution_lib.num_processes() > 1
            and not isinstance(self.data_adapter, TFDatasetAdapter)
        ):
            return self._get_distributed_iterator(distribution)
        return self.data_adapter.get_numpy_iterator()

    def _get_distributed_iterator(self, distribution):
        """Shard each batch so every process trains on its own slice."""

        def shard(d, layout):
            if d is None:
                return None
            return mlx_distribution_lib.distribute_data_input(
                d, layout, distribution.batch_dim_name
            )

        layouts = None
        for data in self.data_adapter.get_numpy_iterator():
            if layouts is None:

                def get_layout(d):
                    if d is None:
                        return None
                    return distribution.get_data_layout(d.shape)

                layouts = tree.map_structure(get_layout, data)
            yield tree.map_structure(shard, data, layouts)
