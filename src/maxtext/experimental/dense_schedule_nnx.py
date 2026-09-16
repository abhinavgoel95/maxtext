"""Read-only adapter from a MaxText scanned Llama stack to the schedule core."""

from flax import nnx
import jax
import jax.numpy as jnp

from maxtext.common.common_types import DecoderBlockType, MODEL_MODE_TRAIN, ShardMode
from maxtext.models.llama2 import LlamaDecoderLayer
from maxtext.utils import maxtext_utils, maxtext_utils_nnx


def validate_config(config):
  """Reject modes whose state or layer structure this first experiment omits."""
  if config.decoder_block != DecoderBlockType.LLAMA2:
    raise ValueError("The first dense-schedule experiment supports decoder_block=llama2 only")
  if not config.scan_layers or config.inhomogeneous_layer_cycle_interval != 1:
    raise ValueError("Use scan_layers=true and inhomogeneous_layer_cycle_interval=1")
  if config.num_decoder_layers < 1:
    raise ValueError("At least one decoder layer is required")
  if config.shard_mode != ShardMode.AUTO:
    raise ValueError("The first dense-schedule experiment requires shard_mode=auto")
  if config.remat_policy not in ("none", "full"):
    raise ValueError("Use remat_policy=none or remat_policy=full for this experiment")
  if config.dropout_rate != 0:
    raise ValueError("Use dropout_rate=0; layer state is read-only during this experiment")
  if config.quantization or config.num_experts != 1 or config.mtp_num_layers != 0:
    raise ValueError("Use unquantized dense layers: quantization='', num_experts=1, mtp_num_layers=0")
  unsupported = (
      "use_qwix_quantization", "use_manual_quantization", "quantize_kvcache",
      "record_internal_nn_metrics", "parameter_memory_host_offload",
      "using_pipeline_parallelism", "use_batch_split_schedule", "te_moe_block",
      "use_multimodal", "use_audio", "learn_to_init_mode",
  )
  enabled = [name for name in unsupported if getattr(config, name, False)]
  if getattr(getattr(config, "lora", None), "enable_lora", False):
    enabled.append("lora.enable_lora")
  if enabled:
    raise ValueError(f"Unsupported in the initial dense-schedule experiment: {', '.join(enabled)}")


def make_layer_adapter(layers, config):
  """Return (layer_apply, params, state) for a fully initialized dense stack.

  Params and state have a leading layer axis for lax.scan. Returned gradients
  from the schedule core KEEP this leading axis: they are profiling outputs,
  not a TrainState update. Parameter metadata is kept for MaxText sharding.
  The caller must initialize in TRAIN mode and trace under the configured mesh
  and logical-axis rules. This adapter does not initialize or update the model.
  """
  validate_config(config)
  if not isinstance(layers, LlamaDecoderLayer):
    raise TypeError("Expected a scanned LlamaDecoderLayer, not a full model or sequential layer list")
  graphdef, params, state = nnx.split(layers, nnx.Param, ...)
  if config.param_scan_axis != 0:
    params = jax.tree.map(lambda x: jnp.moveaxis(x, config.param_scan_axis, 0), params)
  params = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(params, config.num_decoder_layers)
  state = maxtext_utils_nnx.nnx_ensure_scan_leading_axis(state, config.num_decoder_layers)

  def layer_apply(weights, hidden, layer_state, positions, segments):
    # lax.scan slices the values, but their metadata still names the layer axis.
    weights, layer_state = maxtext_utils_nnx.nnx_remove_scan_axis((weights, layer_state), "layers")
    layer = nnx.merge(graphdef, weights, layer_state, copy=True)
    hidden, _ = layer(
        hidden,
        decoder_segment_ids=segments,
        decoder_positions=positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    return hidden

  # Do not confuse policy=None (full remat) with remat being disabled.
  if config.remat_policy == "full":
    layer_apply = jax.checkpoint(
        layer_apply,
        policy=jax.checkpoint_policies.nothing_saveable,
        prevent_cse=maxtext_utils.should_prevent_cse_in_remat(config),
    )
  return layer_apply, params, state
