"""Multi-process data-parallel correctness check for the MLX backend.

Run with `mlx.launch` so each rank is its own process (NOT via pytest):

    KERAS_BACKEND=mlx .venv/bin/mlx.launch --backend ring -n 2 \
        --hosts 127.0.0.1 integration_tests/mlx_data_parallel.py

It asserts that replicated weights stay in sync through an all-reduced
gradient step, that a 2-process half-batch step equals a 1-process
full-batch step, and that `model.fit` under `DataParallel` automatically
shards the full dataset to reproduce the manual-shard reference step.
Each rank exits nonzero on failure; mlx.launch propagates it.
"""

import sys

import mlx.core as mx
import numpy as np

import keras


def _check(name, ok, rank):
    status = "OK" if ok else "FAIL"
    print(f"[rank {rank}] {name}: {status}", flush=True)
    if not ok:
        sys.exit(1)


def _all_equal_across_ranks(x, group):
    """True iff `x` is bit-identical on every rank (range of values is 0)."""
    hi = mx.distributed.all_max(x, group=group)
    lo = mx.distributed.all_min(x, group=group)
    mx.eval(hi, lo)
    return bool(mx.all(hi == lo).item())


def main():
    group = mx.distributed.init(backend="ring")
    rank, world = group.rank(), group.size()
    if world < 2:
        print("Need world_size>=2; launch with mlx.launch -n 2.", flush=True)
        sys.exit(1)

    # Deterministic tiny linear model y = w.x + b with replicated weights
    # (same init on every rank, independent of rank).
    w = mx.array([0.5, -0.25, 1.0, 0.0], dtype=mx.float32)
    b = mx.array(0.1, dtype=mx.float32)

    _check(
        "weights_replicated_at_init",
        _all_equal_across_ranks(w, group) and _all_equal_across_ranks(b, group),
        rank,
    )

    # Full global batch, deterministically generated, identical on all ranks.
    rng = np.random.default_rng(1234)
    n_per_rank = 8
    global_n = n_per_rank * world
    x_full = mx.array(rng.standard_normal((global_n, 4)).astype(np.float32))
    y_full = mx.array(rng.standard_normal((global_n,)).astype(np.float32))

    def mse_grads(xb, yb, w, b):
        def loss_fn(w, b):
            pred = xb @ w + b
            return mx.mean((pred - yb) ** 2)

        return mx.value_and_grad(loss_fn, argnums=(0, 1))(w, b)

    # Data-parallel step: each rank takes its contiguous shard.
    shard = slice(rank * n_per_rank, (rank + 1) * n_per_rank)
    xb, yb = x_full[shard], y_full[shard]
    (loss_local, (gw_local, gb_local)) = mse_grads(xb, yb, w, b)

    # All-reduce + average the grads (the op injected in train_step between
    # grad_fn() and optimizer.apply() when world>1).
    gw = mx.distributed.all_sum(gw_local, group=group) / world
    gb = mx.distributed.all_sum(gb_local, group=group) / world
    mx.eval(gw, gb)

    _check(
        "averaged_grads_identical_across_ranks",
        _all_equal_across_ranks(gw, group)
        and _all_equal_across_ranks(gb, group),
        rank,
    )

    # One SGD step: replicated update from replicated grads stays replicated.
    lr = 0.1
    w_new = w - lr * gw
    b_new = b - lr * gb
    mx.eval(w_new, b_new)

    _check(
        "final_weights_identical_across_ranks",
        _all_equal_across_ranks(w_new, group)
        and _all_equal_across_ranks(b_new, group),
        rank,
    )

    # 2-proc half-batch data-parallel == 1-proc full-batch (within tol).
    # The MSE gradient is the mean over examples, so averaging equal-sized
    # per-shard mean-grads equals the full-batch mean-grad.
    (_, (gw_full, gb_full)) = mse_grads(x_full, y_full, w, b)
    mx.eval(gw_full, gb_full)
    tol = 1e-5
    same_grad = bool(mx.all(mx.abs(gw - gw_full) < tol).item()) and bool(
        mx.abs(gb - gb_full).item() < tol
    )
    _check("dp_halfbatch_equals_single_fullbatch", same_grad, rank)

    # model.fit on the FULL dataset per process: the trainer must shard each
    # batch automatically under DataParallel, so one fit step on the global
    # batch reproduces the manual-shard SGD step computed above.
    distribution = keras.distribution.DataParallel()
    with distribution.scope():
        model = keras.Sequential([keras.layers.Dense(1)])
        model.build((None, 4))
        model.set_weights([np.array(w).reshape(4, 1), np.array(b).reshape(1)])
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=lr), loss="mse"
        )
        model.fit(
            np.array(x_full),
            np.array(y_full).reshape(-1, 1),
            batch_size=global_n,
            epochs=1,
            shuffle=False,
            verbose=0,
        )
    kernel, bias = model.get_weights()

    _check(
        "fit_weights_identical_across_ranks",
        _all_equal_across_ranks(mx.array(kernel), group)
        and _all_equal_across_ranks(mx.array(bias), group),
        rank,
    )

    fit_matches_manual = bool(
        np.all(np.abs(kernel[:, 0] - np.array(w_new)) < tol)
    ) and bool(abs(bias[0] - float(b_new.item())) < tol)
    _check("fit_autoshard_equals_manual_shard_step", fit_matches_manual, rank)

    print(f"[rank {rank}] ALL CHECKS PASSED (world={world})", flush=True)


if __name__ == "__main__":
    main()
