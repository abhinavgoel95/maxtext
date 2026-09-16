"""Small CPU tests for true-loss scheduling, gradient bookkeeping, and fusion."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.experimental.dense_training_schedule import make_training_schedule


def prefix_apply(params, data):
  return params["embedding"][data["inputs"]] * params["prefix_scale"]


def layer_apply(params, hidden, state, positions, segments):
  del positions, segments
  return jnp.tanh(hidden @ params["kernel"] + params["bias"]) * state["gain"]


def loss_apply(params, hidden, data):
  # Tied input/output embedding exercises shared boundary-parameter gradients.
  logits = hidden @ params["embedding"].T + params["head_bias"]
  logprobs = jax.nn.log_softmax(logits)
  token_loss = -jnp.take_along_axis(logprobs, data["targets"][..., None], axis=-1)[..., 0]
  mask = data["targets_segmentation"] != 0
  loss = jnp.sum(jnp.where(mask, token_loss, 0))
  return loss, {"total_loss": loss, "total_weights": jnp.sum(mask)}


def make_inputs(layers, microbatches):
  keys = jax.random.split(jax.random.key(17), 5)
  layer_params = {
      "kernel": jax.random.normal(keys[0], (layers, 4, 4)) * 0.2,
      "bias": jax.random.normal(keys[1], (layers, 4)) * 0.1,
  }
  layer_state = {"gain": jnp.full((layers, 4), 0.9)}
  boundary_params = {
      "embedding": jax.random.normal(keys[2], (7, 4)) * 0.2,
      "prefix_scale": jnp.full((4,), 1.1),
      "head_bias": jnp.arange(7, dtype=jnp.float32) * 0.01,
  }
  data_shape = (microbatches, 2, 3)
  data = {
      "inputs": jax.random.randint(keys[3], data_shape, 0, 7),
      "inputs_position": jnp.broadcast_to(jnp.arange(3), data_shape),
      "inputs_segmentation": jnp.ones(data_shape, dtype=jnp.int32),
      "targets": jax.random.randint(keys[4], data_shape, 0, 7),
      # Unequal valid-token counts exercise unnormalized accumulation.
      "targets_segmentation": (jnp.arange(microbatches * 6).reshape(data_shape) % 3 != 0).astype(jnp.int32),
  }
  return layer_params, layer_state, boundary_params, data


def reference_loss(layer_params, boundary_params, layer_state, data):
  loss_total = jnp.float32(0)
  weights_total = jnp.int32(0)
  for microbatch in range(data["inputs"].shape[0]):
    one = jax.tree.map(lambda x: x[microbatch], data)
    hidden = prefix_apply(boundary_params, one)
    for layer in range(layer_params["kernel"].shape[0]):
      hidden = layer_apply(
          jax.tree.map(lambda x: x[layer], layer_params),
          hidden,
          jax.tree.map(lambda x: x[layer], layer_state),
          one["inputs_position"],
          one["inputs_segmentation"],
      )
    loss, aux = loss_apply(boundary_params, hidden, one)
    loss_total += loss
    weights_total += aux["total_weights"]
  return loss_total, {"total_loss": loss_total, "total_weights": weights_total}


class DenseTrainingScheduleTest(unittest.TestCase):
  def assert_tree_allclose(self, actual, expected):
    self.assertEqual(jax.tree.structure(actual), jax.tree.structure(expected))
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
      np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=2e-5, atol=2e-6)

  def test_loss_and_all_gradients_match_full_autodiff(self):
    for layers in (1, 3):
      for microbatches in (1, 2, 3):
        args = make_inputs(layers, microbatches)
        params, state, boundary, data = args
        (loss, aux), (layer_grads, boundary_grads) = jax.value_and_grad(reference_loss, argnums=(0, 1), has_aux=True)(
            params, boundary, state, data
        )
        for schedule in ("serial", "dual_pipe"):
          with self.subTest(layers=layers, microbatches=microbatches, schedule=schedule):
            step = make_training_schedule(prefix_apply, layer_apply, loss_apply, schedule)
            actual = jax.jit(step)(*args)
            self.assert_tree_allclose(actual, (loss, aux, layer_grads, boundary_grads))

  def test_empty_mask_produces_zero_sums(self):
    params, state, boundary, data = make_inputs(3, 3)
    data["targets_segmentation"] = jnp.zeros_like(data["targets_segmentation"])
    step = make_training_schedule(prefix_apply, layer_apply, loss_apply)
    result = jax.jit(step)(params, state, boundary, data)
    for leaf in jax.tree.leaves(result):
      np.testing.assert_array_equal(leaf, jnp.zeros_like(leaf))

  def test_full_remat_matches_full_autodiff(self):
    args = make_inputs(3, 3)
    params, state, boundary, data = args
    (loss, aux), (layer_grads, boundary_grads) = jax.value_and_grad(reference_loss, argnums=(0, 1), has_aux=True)(
        params, boundary, state, data
    )
    rematted_layer = jax.checkpoint(layer_apply, policy=jax.checkpoint_policies.nothing_saveable)
    step = make_training_schedule(prefix_apply, rematted_layer, loss_apply)
    self.assert_tree_allclose(jax.jit(step)(*args), (loss, aux, layer_grads, boundary_grads))

  def test_bfloat16_gradients_accumulate_in_float32(self):
    params, state, boundary, data = make_inputs(3, 3)
    params, state, boundary = jax.tree.map(lambda x: x.astype(jnp.bfloat16), (params, state, boundary))
    step = make_training_schedule(prefix_apply, layer_apply, loss_apply)
    _, _, layer_grads, boundary_grads = jax.jit(step)(params, state, boundary, data)
    for leaf in jax.tree.leaves((layer_grads, boundary_grads)):
      self.assertEqual(leaf.dtype, jnp.float32)

  def test_backward_and_forward_share_inner_scan(self):
    step = make_training_schedule(prefix_apply, layer_apply, loss_apply)
    graph = jax.make_jaxpr(step)(*make_inputs(3, 3))

    def scans(jaxpr):
      for equation in jaxpr.eqns:
        if equation.primitive.name == "scan":
          body = equation.params["jaxpr"].jaxpr
          yield equation.params["length"], body
          yield from scans(body)

    steady_bodies = [body for length, body in scans(graph.jaxpr) if length == 2]
    self.assertEqual(len(steady_bodies), 1)
    layer_bodies = [body for length, body in scans(steady_bodies[0]) if length == 3]
    self.assertEqual(len(layer_bodies), 1)
    scopes = [str(eq.source_info.name_stack).split("/") for eq in layer_bodies[0].eqns]
    self.assertTrue(any("backward" in scope for scope in scopes))
    self.assertTrue(any("forward" in scope for scope in scopes))


if __name__ == "__main__":
  unittest.main()
