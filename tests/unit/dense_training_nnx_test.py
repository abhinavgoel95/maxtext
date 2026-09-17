"""Tiny NNX integration tests without loading the full MaxText model stack."""

import unittest
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.common.common_types import DecoderBlockType, ShardMode
from maxtext.experimental.dense_training_nnx import (
    _layer_groups, _make_boundaries, _microbatches, _reduce_aux, _restore_gradients,
    _te_layer_metrics, validate_training_config,
)
from maxtext.experimental.dense_training_schedule import make_training_schedule
from maxtext.utils.globals import EPS


class TinyEmbedding(nnx.Module):
  def __init__(self):
    self.embedding = nnx.Param(jax.random.normal(jax.random.key(0), (7, 4)) * 0.2, out_sharding=("vocab", "embed"))


class TinyLayerStack(nnx.Module):
  def __init__(self, layers, scan_axis):
    kernel = jax.random.normal(jax.random.key(1), (layers, 4, 4)) * 0.2
    bias = jax.random.normal(jax.random.key(2), (layers, 4)) * 0.1
    kernel_axes = ["layers", "embed", "mlp"]
    bias_axes = ["layers", "embed"]
    kernel_axes.insert(scan_axis, kernel_axes.pop(0))
    bias_axes.insert(scan_axis, bias_axes.pop(0))
    self.kernel = nnx.Param(jnp.moveaxis(kernel, 0, scan_axis), out_sharding=tuple(kernel_axes))
    self.bias = nnx.Param(jnp.moveaxis(bias, 0, scan_axis), out_sharding=tuple(bias_axes))
    self.gain = nnx.Variable(jnp.full((layers, 4), 0.9))


class TinyExpertStack(nnx.Module):
  """Different parameter shapes/tree from the dense prefix; no TE kernels."""

  def __init__(self, layers, scan_axis):
    for name, shape, axes in (
        ("gate", (layers, 4, 2), ("moe_layers", "embed", "expert")),
        ("experts", (layers, 2, 4, 4), ("moe_layers", "expert", "embed", "mlp")),
    ):
      value = jax.random.normal(jax.random.key(len(shape)), shape) * 0.15
      axes = list(axes)
      axes.insert(scan_axis, axes.pop(0))
      setattr(self, name, nnx.Param(jnp.moveaxis(value, 0, scan_axis), out_sharding=tuple(axes)))


class TinyDecoder(nnx.Module):
  def __init__(self, layers, scan_axis):
    self.layers = TinyLayerStack(layers, scan_axis)
    self.prefix_scale = nnx.Param(jnp.full((4,), 1.1), out_sharding=("embed",))
    self.head_bias = nnx.Param(jnp.arange(7, dtype=jnp.float32) * 0.01, out_sharding=("vocab",))

  def _apply_embedding(self, token_embedder, inputs, positions, *, deterministic, model_mode):
    del positions, deterministic, model_mode
    return token_embedder.embedding.get_value()[inputs] * self.prefix_scale.get_value()

  def apply_output_head(self, token_embedder, hidden, *, deterministic, model_mode):
    del deterministic, model_mode
    # Input and output share the same embedding parameter.
    return hidden @ token_embedder.embedding.get_value().T + self.head_bias.get_value()


class TinyTransformer(nnx.Module):
  def __init__(self, layers=3, scan_axis=1):
    self.token_embedder = TinyEmbedding()
    self.decoder = TinyDecoder(layers, scan_axis)
    self.mesh = None


def loss_from_logits(logits, data, config, mesh, loss_mask=None):
  del mesh
  logprobs = jax.nn.log_softmax(logits)
  xent = -jnp.take_along_axis(logprobs, data["targets"][..., None], axis=-1)[..., 0]
  z_loss = getattr(config, "z_loss_multiplier", 0.0) * jax.nn.logsumexp(logits, axis=-1) ** 2
  mask = data["targets_segmentation"] != 0 if loss_mask is None else loss_mask
  xent_sum = jnp.sum(jnp.where(mask, xent + z_loss, 0))
  return xent_sum, jnp.sum(jnp.where(mask, z_loss, 0)), jnp.sum(mask)


def layer_apply(params, hidden, state, positions, segments):
  del positions, segments
  return jnp.tanh(hidden @ params["kernel"].get_value() + params["bias"].get_value()) * state["gain"].get_value()


def expert_apply(params, hidden, state, positions, segments):
  del state, positions, segments
  scores = jax.nn.softmax(hidden @ params["gate"].get_value(), axis=-1)
  outputs = jnp.tanh(jnp.einsum("...d,edh->...eh", hidden, params["experts"].get_value()))
  return hidden + jnp.sum(outputs * scores[..., None], axis=-2)


def make_data(microbatches, empty=False):
  shape = (microbatches * 2, 3)
  indices = jnp.arange(np.prod(shape)).reshape(shape)
  return {
      "inputs": (indices + 1) % 7,
      "inputs_position": jnp.broadcast_to(jnp.arange(3), shape),
      "inputs_segmentation": jnp.ones(shape, jnp.int32),
      "targets": (indices + 2) % 7,
      "targets_segmentation": jnp.zeros(shape, jnp.int32) if empty else (indices % 3 != 0).astype(jnp.int32),
  }


class DenseTrainingNnxTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    mesh = jax.sharding.Mesh(
        np.array([jax.devices()[0]]).reshape((1,) * 7),
        ("vocab", "embed", "mlp", "layers", "expert", "dense_layers", "moe_layers"),
        axis_types=(jax.sharding.AxisType.Auto,) * 7,
    )
    self.enterContext(jax.set_mesh(mesh))

  def assert_tree_allclose(self, actual, expected):
    # NNX variable metadata is part of the pytree structure, not just leaf shape.
    self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
      np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=2e-5, atol=2e-6)

  def test_boundaries_leave_original_model_untouched(self):
    model = TinyTransformer()
    original_layers = model.decoder.layers
    original_state = nnx.state(model)
    prefix, head, params = _make_boundaries(model, SimpleNamespace(), loss_from_logits)
    self.assertIs(model.decoder.layers, original_layers)
    self.assertNotIn("layers", params["decoder"])
    self.assert_tree_allclose(nnx.state(model), original_state)
    data = make_data(1)
    hidden = jax.jit(prefix)(params, data)
    loss, aux = jax.jit(head)(params, hidden, data)
    self.assertEqual(hidden.shape, (2, 3, 4))
    self.assertEqual(loss.shape, ())
    self.assertEqual(aux["total_weights"], 4)
    self.assertIs(model.decoder.layers, original_layers)
    self.assert_tree_allclose(nnx.state(model), original_state)

  def test_boundary_normalizes_nonzero_z_loss_metric_once(self):
    model = TinyTransformer()
    config = SimpleNamespace(z_loss_multiplier=0.01)
    prefix, head, params = _make_boundaries(model, config, loss_from_logits)
    for empty in (False, True):
      with self.subTest(empty=empty):
        data = make_data(1, empty=empty)
        hidden = prefix(params, data)
        logits = model.decoder.apply_output_head(model.token_embedder, hidden, deterministic=True, model_mode="train")
        expected_loss, z_loss_sum, count = loss_from_logits(logits, data, config, model.mesh)
        loss, aux = jax.jit(head)(params, hidden, data)
        np.testing.assert_allclose(loss, expected_loss)
        np.testing.assert_allclose(aux["z_loss"], z_loss_sum / (count + EPS))
        np.testing.assert_allclose(aux["xent_sum"], expected_loss)
        self.assertEqual(aux["total_weights"], count)
        if not empty:
          self.assertGreater(float(z_loss_sum), 0.0)
          self.assertGreater(int(count), 1)

  def test_restore_gradients_recovers_axis_tree_and_metadata(self):
    for scan_axis in (0, 1):
      with self.subTest(scan_axis=scan_axis):
        model = TinyTransformer(scan_axis=scan_axis)
        params = nnx.state(model, nnx.Param)
        _, _, boundary = _make_boundaries(model, SimpleNamespace(), loss_from_logits)
        layers = nnx.state(model.decoder.layers, nnx.Param)
        leading_layers = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0) * 2, layers)
        boundary_grads = jax.tree.map(lambda x: x * 2, boundary)
        restored = _restore_gradients(leading_layers, boundary_grads, scan_axis)
        self.assert_tree_allclose(restored, jax.tree.map(lambda x: x * 2, params))
        self.assertEqual(restored["decoder"]["layers"]["kernel"].get_metadata(), params["decoder"]["layers"]["kernel"].get_metadata())

  def test_microbatch_order_matches_normal_ga_and_slices_padding(self):
    data = make_data(3)
    actual = _microbatches(data, count=3, micro_batch_size=1)
    # Normal GA reshapes [B*M, ...] to [B, M, ...] before transposition,
    # so MB_i contains rows i, i+M, ... (not consecutive blocks of B rows).
    expected = jax.tree.map(lambda x: jnp.stack([x[index::3][:1] for index in range(3)]), data)
    self.assert_tree_allclose(actual, expected)

  def test_invalid_batch_layout_is_rejected(self):
    data = make_data(1)
    with self.assertRaisesRegex(ValueError, "divisible"):
      _microbatches(data, count=3, micro_batch_size=1)
    with self.assertRaisesRegex(ValueError, "micro_batch_size"):
      _microbatches(data, count=1, micro_batch_size=3)
    with self.assertRaisesRegex(ValueError, "batch keys"):
      _microbatches({"inputs": data["inputs"]}, count=1, micro_batch_size=1)

  def test_all_parameter_gradients_and_sgd_update_match_full_model(self):
    for microbatches in (1, 3):
      for empty in (False, True):
        with self.subTest(microbatches=microbatches, empty=empty):
          model = TinyTransformer(scan_axis=1)
          graphdef, all_params, all_state = nnx.split(model, nnx.Param, ...)
          prefix, head, boundary_params = _make_boundaries(model, SimpleNamespace(), loss_from_logits)
          _, layer_params, layer_state = nnx.split(model.decoder.layers, nnx.Param, ...)
          layer_params = jax.tree.map(lambda x: jnp.moveaxis(x, 1, 0), layer_params)
          data = _microbatches(make_data(microbatches, empty), count=microbatches, micro_batch_size=2)

          def reference(params):
            local_model = nnx.merge(graphdef, params, all_state, copy=True)
            total_loss, total_weights = jnp.float32(0), jnp.int32(0)
            for mb in range(microbatches):
              batch = jax.tree.map(lambda x: x[mb], data)
              hidden = local_model.decoder._apply_embedding(
                  local_model.token_embedder, batch["inputs"], batch["inputs_position"], deterministic=True, model_mode="train"
              )
              stack = local_model.decoder.layers
              for layer in range(3):
                hidden = jnp.tanh(hidden @ stack.kernel.get_value()[:, layer, :] + stack.bias.get_value()[:, layer])
                hidden *= stack.gain.get_value()[layer]
              logits = local_model.decoder.apply_output_head(
                  local_model.token_embedder, hidden, deterministic=True, model_mode="train"
              )
              loss, _, weights = loss_from_logits(logits, batch, None, None)
              total_loss += loss
              total_weights += weights
            return total_loss / jnp.maximum(total_weights, 1)

          reference_value, reference_grads = jax.jit(jax.value_and_grad(reference))(all_params)
          schedule = make_training_schedule(prefix, layer_apply, head)
          loss_sum, aux, layer_grads, boundary_grads = jax.jit(schedule)(layer_params, layer_state, boundary_params, data)
          grads = _restore_gradients(layer_grads, boundary_grads, param_scan_axis=1)
          denominator = jnp.maximum(aux["total_weights"], 1)
          grads = jax.tree.map(lambda x: x / denominator, grads)
          np.testing.assert_allclose(loss_sum / denominator, reference_value, rtol=2e-5, atol=2e-6)
          self.assert_tree_allclose(grads, reference_grads)
          updated = jax.tree.map(lambda p, g: p - 0.01 * g, all_params, grads)
          expected_updated = jax.tree.map(lambda p, g: p - 0.01 * g, all_params, reference_grads)
          self.assert_tree_allclose(updated, expected_updated)

  def test_deepseek_group_gradients_restore_full_model_tree(self):
    for scan_axis in (0, 1):
      with self.subTest(scan_axis=scan_axis):
        model = TinyTransformer(layers=1, scan_axis=scan_axis)
        model.decoder.dense_layers = model.decoder.layers
        del model.decoder.layers
        model.decoder.moe_layers = TinyExpertStack(3, scan_axis)
        names = ("dense_layers", "moe_layers")
        graphdef, all_params, rest = nnx.split(model, nnx.Param, ...)
        prefix, head, boundary = _make_boundaries(model, SimpleNamespace(), loss_from_logits, names)
        self.assertNotIn("dense_layers", boundary["decoder"])
        self.assertNotIn("moe_layers", boundary["decoder"])
        group_params, group_states = [], []
        for name in names:
          _, params, state = nnx.split(getattr(model.decoder, name), nnx.Param, ...)
          group_params.append(jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), params))
          group_states.append(state)
        data = _microbatches(make_data(3), 3, 2)

        def reference(params):
          total = jnp.float32(0)
          local_model = nnx.merge(graphdef, params, rest, copy=True)
          for mb in range(3):
            batch = jax.tree.map(lambda x: x[mb], data)
            hidden = local_model.decoder._apply_embedding(
                local_model.token_embedder, batch["inputs"], batch["inputs_position"],
                deterministic=True, model_mode="train",
            )
            for name, count, apply in zip(names, (1, 3), (layer_apply, expert_apply)):
              _, weights, state = nnx.split(getattr(local_model.decoder, name), nnx.Param, ...)
              weights = jax.tree.map(lambda x: jnp.moveaxis(x, scan_axis, 0), weights)
              for index in range(count):
                hidden = apply(jax.tree.map(lambda x: x[index], weights), hidden,
                               jax.tree.map(lambda x: x[index], state), None, None)
            logits = local_model.decoder.apply_output_head(
                local_model.token_embedder, hidden, deterministic=True, model_mode="train"
            )
            total += loss_from_logits(logits, batch, None, None)[0]
          return total

        expected_loss, expected_grads = jax.jit(jax.value_and_grad(reference))(all_params)
        checkpointed = tuple(jax.checkpoint(f, policy=jax.checkpoint_policies.nothing_saveable)
                             for f in (layer_apply, expert_apply))
        schedule = make_training_schedule(prefix, checkpointed, head)
        loss, _, group_grads, boundary_grads = jax.jit(schedule)(
            tuple(group_params), tuple(group_states), boundary, data
        )
        grads = _restore_gradients(group_grads, boundary_grads, scan_axis, names)
        np.testing.assert_allclose(loss, expected_loss, rtol=2e-5)
        self.assert_tree_allclose(grads, expected_grads)

  def test_te_metrics_preserve_overflow_and_capacity(self):
    layer = nnx.Module()
    layer.sow(nnx.Intermediate, "te_moe_capacity_overflow", jnp.array([False, True]))
    layer.sow(nnx.Intermediate, "te_moe_total_recv_tokens", jnp.array([5, 13], jnp.int32))
    layer.sow(nnx.Intermediate, "te_moe_recv_capacity_per_rank", jnp.int32(12))
    metrics = _te_layer_metrics(nnx.pop(layer, nnx.Intermediate), required=True)
    neutral = _te_layer_metrics({}, required=False)
    reduced = _reduce_aux(jax.tree.map(lambda a, b: jnp.stack([a, b]), neutral, metrics))
    self.assertTrue(bool(reduced["te_moe_capacity_overflow"]))
    self.assertEqual(int(reduced["te_moe_max_total_recv_tokens"]), 13)
    self.assertEqual(int(reduced["te_moe_recv_capacity_per_rank"]), 12)
    with self.assertRaisesRegex(ValueError, "receive-capacity"):
      _te_layer_metrics({}, required=True)

  def test_deepseek_validation_and_group_layout(self):
    config = SimpleNamespace(
        decoder_block=DecoderBlockType.DEEPSEEK, scan_layers=True, inhomogeneous_layer_cycle_interval=1,
        num_decoder_layers=4, first_num_dense_layers=1, shard_mode=ShardMode.AUTO,
        remat_policy="full", dropout_rate=0, quantization="te_no_quant", num_experts=16, mtp_num_layers=0,
        te_moe_block=True, te_gmm_quantization="te_no_quant", load_balance_loss_weight=0,
        routed_bias=False, routed_bias_update_rate=0, num_vocab_tiling=1,
    )
    validate_training_config(config)
    self.assertEqual(_layer_groups(config), (("dense_layers", 1), ("moe_layers", 3)))
    no_dense = SimpleNamespace(**{**vars(config), "first_num_dense_layers": 0})
    validate_training_config(no_dense)
    self.assertEqual(_layer_groups(no_dense), (("moe_layers", 4),))
    for field, value in (("quantization", "te_mxfp8"), ("te_gmm_quantization", "te_mxfp8"),
                         ("te_moe_block", False), ("load_balance_loss_weight", 0.01),
                         ("routed_bias", True), ("routed_bias_update_rate", 0.01),
                         ("first_num_dense_layers", 4)):
      with self.subTest(field=field), self.assertRaises(ValueError):
        validate_training_config(SimpleNamespace(**{**vars(config), field: value}))


if __name__ == "__main__":
  unittest.main()
