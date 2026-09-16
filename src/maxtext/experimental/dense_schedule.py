"""Decoder-stack scheduling only: no optimizer, loss, or numerical checks."""

import jax
import jax.numpy as jnp


def make_schedule_step(layer_apply, schedule, grad_dtype=jnp.float32):
  """Build an unjitted step; the caller compiles the entire schedule once.

  layer_apply(params, hidden, state, positions, segments) returns hidden.
  Params/state have a leading layer axis; inputs/positions/segments/cotangents
  have a leading microbatch axis. State must be read-only during layer_apply.
  Returns (outputs, summed_parameter_gradients, input_gradients).
  """
  if schedule not in ("serial", "dual_pipe"):
    raise ValueError(f"Unknown schedule: {schedule}")
  if not hasattr(jax, "fwd_and_bwd"):
    raise RuntimeError("This experiment requires JAX with jax.fwd_and_bwd")

  forward, backward = jax.fwd_and_bwd(layer_apply, argnums=(0, 1), jitted=False)
  # Reuse the forward trace so saved VJP metadata matches across both scans.
  # These are inlined helpers, NOT separately dispatched GPU executables.
  forward = jax.jit(forward, inline=True)
  backward = jax.jit(backward, inline=True)

  def step(params, state, inputs, positions, segments, cotangents):
    microbatches = inputs.shape[0]
    if microbatches < 1:
      raise ValueError("At least one microbatch is required")
    zeros = jax.tree.map(lambda p: jnp.zeros(p.shape, grad_dtype), params)

    def accumulate(total, grads):
      return jax.tree.map(lambda a, g: a + g.astype(grad_dtype), total, grads)

    def forward_layers(x, pos, seg):
      def body(hidden, layer_data):
        weights, layer_state = layer_data
        with jax.named_scope("forward"):
          return forward(weights, hidden, layer_state, pos, seg)

      return jax.lax.scan(body, x, (params, state))

    def backward_layers(residuals, dy):
      def body(dhidden, residual):
        with jax.named_scope("backward"):
          dweights, dhidden = backward(residual, dhidden)
        return dhidden, dweights

      # reverse=True visits L-1..0 but stacks dweights in layer order 0..L-1.
      dx, dweights = jax.lax.scan(body, dy, residuals, reverse=True)
      return dweights, dx

    if schedule == "serial" or microbatches == 1:
      def serial_microbatch(total, data):
        x, pos, seg, dy = data
        y, residuals = forward_layers(x, pos, seg)
        grads, dx = backward_layers(residuals, dy)
        return accumulate(total, grads), (y, dx)

      with jax.named_scope("serial_FB"):
        total, (outputs, input_grads) = jax.lax.scan(
            serial_microbatch, zeros, (inputs, positions, segments, cotangents)
        )
      return outputs, total, input_grads

    # Fill: F0. Save one residual per layer for B0.
    with jax.named_scope("fill_F0"):
      first_y, residuals = forward_layers(inputs[0], positions[0], segments[0])

    def backward_forward(carry, data):
      previous_residuals, total = carry
      next_x, next_pos, next_seg, previous_dy = data
      reversed_residuals = jax.tree.map(lambda r: r[::-1], previous_residuals)

      def combined_layer(carry, layer_data):
        dhidden, next_hidden = carry
        old_residual, next_weights, next_state = layer_data

        # B_i at layer L-1-k. Neither branch consumes the other's result.
        with jax.named_scope("backward"):
          dweights, dhidden = backward(old_residual, dhidden)

        # F_(i+1) at layer k, in the SAME scan body / compiled executable.
        with jax.named_scope("forward"):
          next_hidden, new_residual = forward(
              next_weights, next_hidden, next_state, next_pos, next_seg
          )
        return (dhidden, next_hidden), (dweights, new_residual)

      with jax.named_scope("combined_bf_layers"):
        (dx, next_y), (reversed_grads, next_residuals) = jax.lax.scan(
            combined_layer,
            (previous_dy, next_x),
            (reversed_residuals, params, state),
        )
      grads = jax.tree.map(lambda g: g[::-1], reversed_grads)
      return (next_residuals, accumulate(total, grads)), (next_y, dx)

    # Steady state: B0F1, B1F2, ... . Keep only the next microbatch's residuals
    # in the outer carry; accumulate dW rather than storing dW for every MB.
    with jax.named_scope("steady_Bi_Fnext"):
      (residuals, total), (later_y, earlier_dx) = jax.lax.scan(
          backward_forward,
          (residuals, zeros),
          (inputs[1:], positions[1:], segments[1:], cotangents[:-1]),
      )

    # Drain: B_(M-1).
    with jax.named_scope("drain_Blast"):
      last_grads, last_dx = backward_layers(residuals, cotangents[-1])
    outputs = jnp.concatenate((first_y[None], later_y), axis=0)
    input_grads = jnp.concatenate((earlier_dx, last_dx[None]), axis=0)
    return outputs, accumulate(total, last_grads), input_grads

  return step
