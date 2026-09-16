"""Profile serial or fused-BF execution of MaxText's dense decoder layers."""

import argparse
from pathlib import Path
import sys
import time

from flax import linen as nn
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from maxtext.common.profiler import Profiler
from maxtext.configs import pyconfig
from maxtext.experimental.dense_schedule import make_schedule_step
from maxtext.experimental.dense_schedule_nnx import make_layer_adapter, validate_config
from maxtext.utils import maxtext_utils, maxtext_utils_nnx, model_creation_utils


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--schedule", choices=("serial", "dual_pipe"), default="dual_pipe")
  parser.add_argument("--output-dir", type=Path, default=Path("dense_schedule_runs"))
  parser.add_argument("config", help="Normal MaxText YAML; experiment flags go before this")
  parser.add_argument("overrides", nargs=argparse.REMAINDER, help="MaxText key=value overrides")
  args = parser.parse_args()

  # MaxText config initialization handles distributed setup. Do not initialize
  # a JAX device/backend before this call.
  config = pyconfig.initialize([sys.argv[0], args.config, *args.overrides])
  validate_config(config)
  mesh = maxtext_utils.get_mesh_from_config(config)
  microbatches = config.gradient_accumulation_steps
  batch = config.global_batch_size_to_train_on // microbatches
  if batch < 1 or config.global_batch_size_to_train_on % microbatches:
    raise ValueError("Global batch must be divisible by gradient_accumulation_steps")
  if config.steps < 1:
    raise ValueError("steps must be positive")
  profiler = Profiler(config)  # Fail invalid capture-window settings before compile.

  with jax.set_mesh(mesh), nn.logical_axis_rules(config.logical_axis_rules):
    create_model, abstract_model = model_creation_utils.create_nnx_abstract_model(config, mesh=mesh)
    # Return/init just the stack, not embeddings, output head, or optimizer.
    # MaxText's helper preserves the configured parameter shardings.
    layers = maxtext_utils_nnx.create_nnx_sharded_model(
        abstract_model.decoder.layers, lambda: create_model().decoder.layers, mesh=mesh
    )
    layer_apply, params, state = make_layer_adapter(layers, config)

    hidden_sharding = nn.logical_to_mesh_sharding(
        P(None, "activation_batch", "activation_length", "activation_embed"), mesh
    )
    index_sharding = nn.logical_to_mesh_sharding(P(None, "activation_batch", "activation_length"), mesh)
    shape = (microbatches, batch, config.max_target_length, config.emb_dim)

    def initialize_inputs(key):
      x_key, dy_key = jax.random.split(key)
      # Generate directly in microbatch layout, with identical seeds for both
      # schedules. These are output cotangents, not a training loss.
      x = jax.random.normal(x_key, shape, config.dtype)
      dy = jax.random.normal(dy_key, shape, config.dtype)
      pos = jnp.broadcast_to(jnp.arange(shape[2], dtype=jnp.int32), shape[:-1])
      seg = jnp.ones(shape[:-1], dtype=jnp.int32)
      return x, pos, seg, dy

    inputs, positions, segments, cotangents = jax.jit(
        initialize_inputs,
        out_shardings=(hidden_sharding, index_sharding, index_sharding, hidden_sharding),
    )(jax.random.key(0))

    step = make_schedule_step(layer_apply, args.schedule, config.grad_dtype)
    params_sharding = jax.tree.map(lambda x: x.sharding, params)
    state_sharding = jax.tree.map(lambda x: x.sharding, state)
    step = jax.jit(
        step,
        in_shardings=(params_sharding, state_sharding, hidden_sharding,
                      index_sharding, index_sharding, hidden_sharding),
        out_shardings=(hidden_sharding, params_sharding, hidden_sharding),
    )
    step_args = (params, state, inputs, positions, segments, cotangents)
    jax.block_until_ready(step_args)
    compiled = step.lower(*step_args).compile()
    result = jax.block_until_ready(compiled(*step_args))  # Warmup outside capture.

    output_dir = args.output_dir / config.run_name / args.schedule
    output_dir.mkdir(parents=True, exist_ok=True)
    # No reliance on MaxText's default jit_train_step HLO-name filter.
    (output_dir / f"rank_{jax.process_index()}_after_optimizations.hlo.txt").write_text(
        compiled.as_text(), encoding="utf-8"
    )

    multihost_utils.sync_global_devices("dense_schedule_start")
    start = time.perf_counter()
    for iteration in range(config.steps):
      profiler.maybe_activate_profiler(iteration, result)
      with jax.profiler.StepTraceAnnotation(args.schedule, step_num=iteration):
        # This is the ONLY model executable dispatched per measured iteration.
        result = jax.block_until_ready(compiled(*step_args))
      profiler.maybe_deactivate_profiler(iteration, result)
    elapsed = time.perf_counter() - start
    print(
        f"COMPLETED rank={jax.process_index()} schedule={args.schedule} "
        f"layers={config.num_decoder_layers} microbatches={microbatches} "
        f"wall_seconds_per_step={elapsed / config.steps:.6f} "
        f"hlo={output_dir} numerical_correctness=not_checked",
        flush=True,
    )


if __name__ == "__main__":
  main()
