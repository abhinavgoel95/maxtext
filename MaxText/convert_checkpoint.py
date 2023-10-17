"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more
import jax
import os
import sys

jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS","") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
print(f"Found {jax.device_count()} devices.")

from typing import Sequence
import datetime
from absl import app
from flax.linen import partitioning as nn_partitioning
from flax import linen as nn
import numpy as np
import optax
from tensorboardX import SummaryWriter
from flax.training import train_state
from layers import Transformer
import pyconfig
from input_pipeline import create_data_iterator_with_tokenizer
import max_utils
import checkpointing

import jax.numpy as jnp
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

from jax.experimental.compilation_cache import compilation_cache as cc

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

import max_logging
cc.initialize_cache(os.path.expanduser("~/jax_cache"))



def calculate_num_params_from_pytree(params):
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  return total_parameters


def to_int8(t):
    return jax.tree_map(lambda x:x.astype(jnp.int8) if x.dtype == jnp.float32 else x, t)


def convert_checkpoint(config):
  """Main Training loop.
  Args:
    config:
  Returns:

  """
  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      config.checkpoint_dir,
      config.enable_checkpointing,
      config.async_checkpointing,
      config.save_period,
  )
  # Initial PRNG Keys
  init_rng, nextrng = random.split(random.PRNGKey(config.init_weights_seed), 2)

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Model and Optimizer definition
  model = Transformer(config, mesh)

  # We use AdamW following Llama2's training details, see https://arxiv.org/pdf/2307.09288.pdf section 2.2
  tx = optax.adamw(
      max_utils.create_learning_rate_schedule(config),
      b1=config.adam_b1,
      b2=config.adam_b2,
      eps=config.adam_eps,
      eps_root=config.adam_eps_root,
      weight_decay=config.adam_weight_decay,
  )

  state, state_mesh_annotations = max_utils.setup_initial_state(model, tx, config, init_rng, mesh, checkpoint_manager)

  num_model_parameters = calculate_num_params_from_pytree(state.params)
  max_logging.log(f"number parameters: {num_model_parameters/10**9:.3f} billion")

  step = 0
  with jax.spmd_mode('allow_all'):
    if config.convert_int8:
      inference_state = train_state.TrainState(
        step=step,
        apply_fn=state.apply_fn,
        params=to_int8(state.params),
        tx=None,
        opt_state={}
        )
    else:
      inference_state = train_state.TrainState(
        step=step,
        apply_fn=state.apply_fn,
        params=state.params,
        tx=None,
        opt_state={}
        )


  if checkpoint_manager is not None:
    if checkpoint_manager.save(step, inference_state):
      max_logging.log(f"saved a checkpoint at step {step}")
    # Upon preemption, exit when and only when all ongoing saves are complete.
    if checkpoint_manager.reached_preemption(step):
      checkpoint_manager.wait_until_finished()
      sys.exit()

  return inference_state

def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  os.environ["TFDS_DATA_DIR"] = pyconfig.config.dataset_path
  convert_checkpoint(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
