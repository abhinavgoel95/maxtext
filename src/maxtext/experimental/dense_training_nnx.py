"""Opt-in dense 1F1B gradient accumulation for the normal MaxText train step.

The decoder's backward/next-forward layer pair shares one scan body. Embedding
and loss/head differentiation happen at its boundaries, with fixed parameters
throughout the accumulated step. There is no optimizer or host dispatch here.
"""

from flax import nnx
import jax
import jax.numpy as jnp

from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.experimental.dense_training_schedule import make_training_schedule


def validate_training_config(config):
  # Import the full MaxText model dependencies only for this opt-in path.
  from maxtext.experimental.dense_schedule_nnx import validate_config

  validate_config(config)
  if getattr(config, "training_objective", "causal_lm") != "causal_lm":
    raise ValueError("dual_pipe currently supports training_objective=causal_lm only")
  if getattr(config, "attention_type", "global") == "block_diffusion":
    raise ValueError("dual_pipe does not support block-diffusion attention with causal-LM targets")
  if config.num_vocab_tiling != 1:
    raise ValueError("dual_pipe currently requires num_vocab_tiling=1")
  if getattr(config, "mhc_expansion_rate", 1) != 1:
    raise ValueError("dual_pipe currently requires mhc_expansion_rate=1")
  unsupported = (
      "use_tunix_gradient_accumulation", "shard_optimizer_over_data",
      "optimizer_memory_host_offload", "use_indexer", "enable_diloco",
      "routed_bias", "retry_when_tokens_dropped", "use_qk_clip", "engram_layers",
  )
  enabled = [name for name in unsupported if getattr(config, name, False)]
  if enabled:
    raise ValueError(f"Unsupported with gradient_accumulation_schedule=dual_pipe: {', '.join(enabled)}")
  if not hasattr(jax, "fwd_and_bwd"):
    raise RuntimeError("dual_pipe requires a JAX build providing jax.fwd_and_bwd")


def _make_boundaries(model, config, causal_lm_loss):
  """Reuse MaxText's actual embedding/head methods without carrying layer weights.

  The copy changes only Python graph structure, not the caller's model. Both
  boundaries share the same parameter tree, so tied embedding gradients add.
  Zero-dropout, unquantized execution does not advance mutable model state.
  """
  boundary_model = nnx.clone(model)
  del boundary_model.decoder.layers
  graphdef, params, state = nnx.split(boundary_model, nnx.Param, ...)

  def prefix_apply(weights, batch):
    local_model = nnx.merge(graphdef, weights, state, copy=True)
    return local_model.decoder._apply_embedding(
        local_model.token_embedder,
        batch["inputs"],
        batch["inputs_position"],
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

  def loss_apply(weights, hidden, batch):
    local_model = nnx.merge(graphdef, weights, state, copy=True)
    logits = local_model.decoder.apply_output_head(
        local_model.token_embedder, hidden, deterministic=True, model_mode=MODEL_MODE_TRAIN
    )
    xent_sum, z_loss, total_weights = causal_lm_loss(logits, batch, config, local_model.mesh)
    return xent_sum, {"xent_sum": xent_sum, "z_loss": z_loss, "total_weights": total_weights}

  return prefix_apply, loss_apply, params


def _microbatches(data, count, micro_batch_size):
  """Keep the exact interleaved batch layout and per-MB slicing used by GA."""
  required = {"inputs", "inputs_position", "inputs_segmentation", "targets", "targets_segmentation"}
  if not required.issubset(data):
    raise ValueError(f"dual_pipe requires batch keys: {sorted(required - data.keys())}")
  if "forced_routed_experts" in data:
    raise ValueError("dual_pipe does not support forced expert routing")

  def reshape(value):
    if value.ndim < 1 or value.shape[0] % count:
      raise ValueError("The leading data dimension must be divisible by gradient_accumulation_steps")
    batch_size = value.shape[0] // count
    if not 0 < micro_batch_size <= batch_size:
      raise ValueError("Invalid micro_batch_size_to_train_on for dual_pipe")
    value = value.reshape((batch_size, count) + value.shape[1:])
    return jnp.swapaxes(value, 0, 1)[:, :micro_batch_size]

  return jax.tree.map(reshape, data)


def _restore_gradients(layer_grads, boundary_grads, param_scan_axis):
  """Restore the full model parameter tree and its original layer axis."""
  if param_scan_axis != 0:
    layer_grads = jax.tree.map(lambda x: jnp.moveaxis(x, 0, param_scan_axis), layer_grads)
  return nnx.merge_state(boundary_grads, nnx.State({"decoder": {"layers": layer_grads}}))


def dualpipe_loss_and_grad(config, model, params_shardings, data, causal_lm_loss):
  """Drop-in loss/aux/gradient result for train_step, not a separate executable."""
  from maxtext.experimental.dense_schedule_nnx import make_layer_adapter
  from maxtext.utils.sharding import maybe_shard_with_name

  validate_training_config(config)
  if not isinstance(model, nnx.Module):
    raise TypeError("dual_pipe requires an NNX model")

  # Use the same input and output parameter sharding constraints as normal GA.
  # Work on a private graph so the optimizer still sees its original state.
  local_model = nnx.clone(model)
  nnx.pop(local_model, nnx.Intermediate)
  params = nnx.state(local_model, nnx.Param)
  supported_dtypes = (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
  if any(p.dtype not in supported_dtypes for p in jax.tree.leaves(params)):
    raise ValueError("dual_pipe currently supports float32/bfloat16/float16 parameters, not quantized weights")

  def shard(value, spec):
    return maybe_shard_with_name(value, spec, config.shard_mode, debug_sharding=config.debug_sharding)

  params = jax.tree.map(shard, params, params_shardings)
  nnx.update(local_model, params)
  layer_apply, layer_params, layer_state = make_layer_adapter(local_model.decoder.layers, config)
  prefix_apply, loss_apply, boundary_params = _make_boundaries(local_model, config, causal_lm_loss)
  batch = _microbatches(data, config.gradient_accumulation_steps, config.micro_batch_size_to_train_on)
  schedule = make_training_schedule(prefix_apply, layer_apply, loss_apply, grad_dtype=config.grad_dtype)
  with jax.named_scope("dual_pipe"):
    loss_sum, aux, layer_grads, boundary_grads = schedule(layer_params, layer_state, boundary_params, batch)

  raw_grads = _restore_gradients(layer_grads, boundary_grads, config.param_scan_axis)
  if jax.tree.structure(raw_grads) != jax.tree.structure(params):
    raise ValueError("dual_pipe gradient tree does not match the full model parameter tree")
  raw_grads = jax.tree.map(shard, raw_grads, params_shardings)

  # Match GA's valid-token normalization, including completely padded batches.
  # The optimizer receives every parameter gradient, including embeddings/head.
  has_weights = aux["total_weights"] > 0
  denominator = jnp.maximum(aux["total_weights"], 1)
  loss = jnp.where(has_weights, loss_sum / denominator, 0.0)
  raw_grads = jax.tree.map(
      lambda x: jnp.where(has_weights, x / denominator, jnp.zeros_like(x)), raw_grads
  )
  aux.update(
      intermediate_outputs={}, moe_lb_loss=0.0, indexer_loss=0.0, mtp_loss=0.0,
      moe_bias_updates=None, mtp_moe_bias_updates=None, batch_stats=None, has_moe_overflow=jnp.bool_(False),
  )
  return loss, aux, raw_grads
