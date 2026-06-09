import mlx.core as mx

from keras.src import tree
from keras.src.backend.mlx.core import _flip
from keras.src.backend.mlx.core import convert_to_tensor


def rnn(
    step_function,
    inputs,
    initial_states,
    go_backwards=False,
    mask=None,
    constants=None,
    unroll=False,
    input_length=None,
    time_major=False,
    zero_output_for_mask=False,
    return_all_outputs=True,
):
    def swap_batch_timestep(input_t):
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        return mx.transpose(input_t, axes)

    if not time_major:
        inputs = tree.map_structure(swap_batch_timestep, inputs)

    flattened_inputs = tree.flatten(inputs)
    time_steps = flattened_inputs[0].shape[0]

    if mask is not None:
        if mask.dtype != mx.bool_:
            mask = mask.astype(mx.bool_)
        if len(mask.shape) == 2:
            mask = mx.expand_dims(mask, axis=-1)
        if not time_major:
            mask = swap_batch_timestep(mask)

    if constants is None:
        constants = []

    def _expand_mask(mask_t, input_t, fixed_dim=1):
        if tree.is_nested(mask_t):
            raise ValueError(
                f"mask_t is expected to be tensor, but got {mask_t}"
            )
        if tree.is_nested(input_t):
            raise ValueError(
                f"input_t is expected to be tensor, but got {input_t}"
            )
        rank_diff = len(input_t.shape) - len(mask_t.shape)
        for _ in range(rank_diff):
            mask_t = mx.expand_dims(mask_t, axis=-1)
        multiples = [1] * fixed_dim + list(input_t.shape[fixed_dim:])
        return mx.tile(mask_t, multiples)

    if unroll:
        if not time_steps:
            raise ValueError("Unrolling requires a fixed number of timesteps.")
        states = tuple(initial_states)
        successive_states = []
        successive_outputs = []

        def _process_single_input_t(input_t):
            input_t = unstack(input_t)
            if go_backwards:
                input_t.reverse()
            return input_t

        if tree.is_nested(inputs):
            processed_input = tree.map_structure(
                _process_single_input_t, inputs
            )
        else:
            processed_input = (_process_single_input_t(inputs),)

        def _get_input_tensor(time):
            inp = [t_[time] for t_ in processed_input]
            return tree.pack_sequence_as(inputs, inp)

        if mask is not None:
            mask_list = unstack(mask)
            if go_backwards:
                mask_list.reverse()

            for i in range(time_steps):
                inp = _get_input_tensor(i)
                mask_t = mask_list[i]
                output, new_states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                tiled_mask_t = _expand_mask(mask_t, output)

                if not successive_outputs:
                    prev_output = mx.zeros_like(output)
                else:
                    prev_output = successive_outputs[-1]

                output = mx.where(tiled_mask_t, output, prev_output)

                flat_states = tree.flatten(states)
                flat_new_states = tree.flatten(new_states)
                tiled_mask_t = tuple(
                    _expand_mask(mask_t, s) for s in flat_states
                )
                flat_final_states = tuple(
                    mx.where(m, s, ps)
                    for m, s, ps in zip(
                        tiled_mask_t, flat_new_states, flat_states
                    )
                )
                states = tree.pack_sequence_as(states, flat_final_states)

                if return_all_outputs:
                    successive_outputs.append(output)
                    successive_states.append(states)
                else:
                    successive_outputs = [output]
                    successive_states = [states]
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = mx.stack(successive_outputs)

        else:  # mask is None
            for i in range(time_steps):
                inp = _get_input_tensor(i)
                output, states = step_function(
                    inp, tuple(states) + tuple(constants)
                )
                if return_all_outputs:
                    successive_outputs.append(output)
                    successive_states.append(states)
                else:
                    successive_outputs = [output]
                    successive_states = [states]
            last_output = successive_outputs[-1]
            new_states = successive_states[-1]
            outputs = mx.stack(successive_outputs)

    else:  # unroll == False
        if mask is not None:

            def _step(states, current_input):
                current_input, current_mask = current_input
                is_masked = mx.all(
                    mx.logical_not(current_mask), axis=-1, keepdims=True
                )

                output_t, new_states = step_function(current_input, states)

                if zero_output_for_mask:
                    masked_outs = mx.where(
                        is_masked, mx.zeros_like(output_t), output_t
                    )
                else:
                    output_tm1 = states[0]
                    if tree.is_nested(output_tm1):
                        output_tm1 = states[-1][0]
                    masked_outs = mx.where(is_masked, output_tm1, output_t)

                new_states = tree.map_structure(
                    lambda s, ns: mx.where(is_masked, s, ns),
                    states,
                    new_states,
                )
                return (new_states, masked_outs)

            scan_xs = (inputs, mask)

        else:

            def _step(states, current_input):
                output_t, new_states = step_function(current_input, states)
                return new_states, output_t

            scan_xs = inputs

        new_states, outputs = mlx_scan(
            f=_step,
            init=initial_states,
            xs=scan_xs,
            reverse=go_backwards,
            mask=mask,
        )

        if go_backwards:
            outputs = _flip(outputs, axis=0)
        last_output = outputs[-1]

    if not time_major:
        outputs = tree.map_structure(swap_batch_timestep, outputs)

    return last_output, outputs, new_states


def lstm(
    inputs,
    initial_state_h,
    initial_state_c,
    mask,
    kernel,
    recurrent_kernel,
    bias,
    activation,
    recurrent_activation,
    return_sequences=False,
    go_backwards=False,
    unroll=False,
):
    # Masking not supported by the fast path; fall back to generic RNN loop.
    if mask is not None:
        raise NotImplementedError

    inputs = convert_to_tensor(inputs)
    initial_state_h = convert_to_tensor(initial_state_h)
    initial_state_c = convert_to_tensor(initial_state_c)
    kernel = convert_to_tensor(kernel)
    recurrent_kernel = convert_to_tensor(recurrent_kernel)
    if bias is not None:
        bias = convert_to_tensor(bias)

    # inputs: (batch, seq_len, input_dim) -> (seq_len, batch, input_dim)
    inputs = mx.transpose(inputs, [1, 0, 2])

    if go_backwards:
        inputs = _flip(inputs, axis=0)

    # Pre-compute input projections: (seq_len, batch, 4*units)
    x_proj = mx.matmul(inputs, kernel)
    if bias is not None:
        x_proj = x_proj + bias

    time_steps = inputs.shape[0]
    h = initial_state_h
    c = initial_state_c
    output_list = []

    for t in range(time_steps):
        z = x_proj[t] + mx.matmul(h, recurrent_kernel)
        z_i, z_f, z_c, z_o = mx.split(z, 4, axis=-1)

        new_c = recurrent_activation(z_f) * c + recurrent_activation(
            z_i
        ) * activation(z_c)
        new_h = recurrent_activation(z_o) * activation(new_c)

        h = new_h
        c = new_c
        output_list.append(h)

    # (seq_len, batch, units) -> (batch, seq_len, units)
    outputs = mx.stack(output_list, axis=0)
    outputs = mx.transpose(outputs, [1, 0, 2])

    last_output = h

    if not return_sequences:
        outputs = mx.expand_dims(last_output, axis=1)

    if go_backwards and return_sequences:
        outputs = _flip(outputs, axis=1)

    return last_output, outputs, [h, c]


def bidirectional_lstm(
    inputs,
    fwd_initial_state_h,
    fwd_initial_state_c,
    bwd_initial_state_h,
    bwd_initial_state_c,
    mask,
    fwd_kernel,
    fwd_recurrent_kernel,
    fwd_bias,
    bwd_kernel,
    bwd_recurrent_kernel,
    bwd_bias,
    activation,
    recurrent_activation,
    return_sequences=False,
    unroll=False,
):
    if mask is not None:
        raise NotImplementedError

    fwd_out = lstm(
        inputs,
        fwd_initial_state_h,
        fwd_initial_state_c,
        None,
        fwd_kernel,
        fwd_recurrent_kernel,
        fwd_bias,
        activation,
        recurrent_activation,
        return_sequences=return_sequences,
        go_backwards=False,
        unroll=unroll,
    )
    bwd_out = lstm(
        inputs,
        bwd_initial_state_h,
        bwd_initial_state_c,
        None,
        bwd_kernel,
        bwd_recurrent_kernel,
        bwd_bias,
        activation,
        recurrent_activation,
        return_sequences=return_sequences,
        go_backwards=True,
        unroll=unroll,
    )
    return fwd_out, bwd_out


def gru(
    inputs,
    initial_state,
    mask,
    kernel,
    recurrent_kernel,
    bias,
    activation,
    recurrent_activation,
    return_sequences=False,
    go_backwards=False,
    unroll=False,
    reset_after=True,
):
    if mask is not None:
        raise NotImplementedError

    inputs = convert_to_tensor(inputs)
    initial_state = convert_to_tensor(initial_state)
    kernel = convert_to_tensor(kernel)
    recurrent_kernel = convert_to_tensor(recurrent_kernel)
    if bias is not None:
        bias = convert_to_tensor(bias)

    hidden_size = recurrent_kernel.shape[0]

    # inputs: (batch, seq_len, input_dim) -> (seq_len, batch, input_dim)
    inputs = mx.transpose(inputs, [1, 0, 2])

    if go_backwards:
        inputs = _flip(inputs, axis=0)

    if reset_after:
        # bias shape: (2, 3*units) — row 0 is input bias, row 1 is recurrent
        if bias is not None:
            input_bias = bias[0]
            recurrent_bias = bias[1]
        else:
            input_bias = mx.zeros((3 * hidden_size,))
            recurrent_bias = mx.zeros((3 * hidden_size,))

        # Pre-compute input projections: (seq_len, batch, 3*units)
        x_all = mx.matmul(inputs, kernel) + input_bias

        time_steps = inputs.shape[0]
        h = initial_state
        output_list = []

        for t in range(time_steps):
            x_t = x_all[t]
            h_all = mx.matmul(h, recurrent_kernel) + recurrent_bias

            x_z, x_r, x_h = mx.split(x_t, 3, axis=-1)
            h_z, h_r, h_h = mx.split(h_all, 3, axis=-1)

            z = recurrent_activation(x_z + h_z)
            r = recurrent_activation(x_r + h_r)
            hh = activation(x_h + r * h_h)

            h_new = z * h + (1 - z) * hh
            h = h_new
            output_list.append(h)
    else:
        # reset_after=False: apply reset gate before recurrent projection
        if bias is not None:
            # bias is a single vector of shape (3*units,)
            input_bias = bias
        else:
            input_bias = mx.zeros((3 * hidden_size,))

        # Pre-compute input projections: (seq_len, batch, 3*units)
        x_all = mx.matmul(inputs, kernel) + input_bias

        # Split recurrent kernel into gate and candidate parts
        rk_z = recurrent_kernel[:, :hidden_size]
        rk_r = recurrent_kernel[:, hidden_size : 2 * hidden_size]
        rk_h = recurrent_kernel[:, 2 * hidden_size :]

        time_steps = inputs.shape[0]
        h = initial_state
        output_list = []

        for t in range(time_steps):
            x_t = x_all[t]
            x_z, x_r, x_h = mx.split(x_t, 3, axis=-1)

            z = recurrent_activation(x_z + mx.matmul(h, rk_z))
            r = recurrent_activation(x_r + mx.matmul(h, rk_r))
            hh = activation(x_h + mx.matmul(r * h, rk_h))

            h_new = z * h + (1 - z) * hh
            h = h_new
            output_list.append(h)

    # (seq_len, batch, units) -> (batch, seq_len, units)
    outputs = mx.stack(output_list, axis=0)
    outputs = mx.transpose(outputs, [1, 0, 2])

    last_output = h

    if not return_sequences:
        outputs = mx.expand_dims(last_output, axis=1)

    if go_backwards and return_sequences:
        outputs = _flip(outputs, axis=1)

    return last_output, outputs, [h]


def bidirectional_gru(
    inputs,
    fwd_initial_state,
    bwd_initial_state,
    mask,
    fwd_kernel,
    fwd_recurrent_kernel,
    fwd_bias,
    bwd_kernel,
    bwd_recurrent_kernel,
    bwd_bias,
    activation,
    recurrent_activation,
    return_sequences=False,
    unroll=False,
    reset_after=True,
):
    if mask is not None:
        raise NotImplementedError

    fwd_out = gru(
        inputs,
        fwd_initial_state,
        None,
        fwd_kernel,
        fwd_recurrent_kernel,
        fwd_bias,
        activation,
        recurrent_activation,
        return_sequences=return_sequences,
        go_backwards=False,
        unroll=unroll,
        reset_after=reset_after,
    )
    bwd_out = gru(
        inputs,
        bwd_initial_state,
        None,
        bwd_kernel,
        bwd_recurrent_kernel,
        bwd_bias,
        activation,
        recurrent_activation,
        return_sequences=return_sequences,
        go_backwards=True,
        unroll=unroll,
        reset_after=reset_after,
    )
    return fwd_out, bwd_out


def unstack(x, axis=0):
    return [mx.take(x, i, axis=axis) for i in range(x.shape[axis])]


def mlx_scan(f, init, xs, reverse=False, mask=None):
    states = init
    outputs = []

    if mask is not None:
        x, mask_tensor = xs
        x = _flip(x, axis=0) if reverse else x
        mask_tensor = _flip(mask_tensor, axis=0) if reverse else mask_tensor

        for i in range(x.shape[0]):
            states, output = f(states, (x[i], mask_tensor[i]))
            outputs.append(output)
    else:
        xs = _flip(xs, axis=0) if reverse else xs

        for i in range(xs.shape[0]):
            states, output = f(states, xs[i])
            outputs.append(output)

    outputs = mx.stack(outputs)

    if reverse:
        outputs = _flip(outputs, axis=0)

    return states, outputs


def cudnn_ok(
    activation,
    recurrent_activation,
    unroll,
    use_bias=True,
):
    return False
