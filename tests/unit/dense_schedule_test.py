"""Structural smoke tests; intentionally not a model-numerics test suite."""

import unittest

import jax
import jax.numpy as jnp

from maxtext.experimental.dense_schedule import make_schedule_step


def tiny_layer(weights, hidden, state, positions, segments):
  del state, positions, segments
  return jnp.tanh(hidden @ weights)


class DenseScheduleTest(unittest.TestCase):
  def test_compile_and_output_structure(self):
    for microbatches in (1, 2, 3):
      for schedule in ("serial", "dual_pipe"):
        with self.subTest(microbatches=microbatches, schedule=schedule):
          weights = jnp.ones((4, 8, 8), dtype=jnp.float32)
          inputs = jnp.ones((microbatches, 2, 3, 8), dtype=jnp.float32)
          positions = jnp.zeros(inputs.shape[:-1], dtype=jnp.int32)
          segments = jnp.ones_like(positions)
          step = make_schedule_step(tiny_layer, schedule)
          args = (weights, {}, inputs, positions, segments, inputs)
          compiled = jax.jit(step).lower(*args).compile()
          outputs, grads, dx = jax.block_until_ready(compiled(*args))
          self.assertEqual(outputs.shape, inputs.shape)
          self.assertEqual(grads.shape, weights.shape)  # No microbatch dW axis.
          self.assertEqual(dx.shape, inputs.shape)

  def test_both_paths_are_in_one_scan_body(self):
    weights = jnp.ones((4, 8, 8))
    inputs = jnp.ones((3, 2, 3, 8))
    positions = jnp.zeros(inputs.shape[:-1], dtype=jnp.int32)
    segments = jnp.ones_like(positions)
    step = make_schedule_step(tiny_layer, "dual_pipe")
    graph = jax.make_jaxpr(step)(weights, {}, inputs, positions, segments, inputs)

    def scan_bodies(jaxpr):
      for equation in jaxpr.eqns:
        if equation.primitive.name == "scan":
          body = equation.params["jaxpr"].jaxpr
          yield equation.params["length"], body
          yield from scan_bodies(body)

    steady = [body for length, body in scan_bodies(graph.jaxpr) if length == 2]
    self.assertEqual(len(steady), 1)  # M-1 steady-state iterations.
    layer_bodies = [body for length, body in scan_bodies(steady[0]) if length == 4]
    self.assertEqual(len(layer_bodies), 1)
    scopes = [str(eq.source_info.name_stack) for eq in layer_bodies[0].eqns]
    self.assertTrue(any("backward" in scope.split("/") for scope in scopes))
    self.assertTrue(any("forward" in scope.split("/") for scope in scopes))


if __name__ == "__main__":
  unittest.main()
