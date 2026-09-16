"""Pure-JAX dense 1F1B gradient accumulation with the real training loss.

The caller owns normalization and the optimizer update and jits the enclosing
training step. This module sums loss, metrics, and gradients over microbatches.
"""

import jax
import jax.numpy as jnp


def make_training_schedule(prefix_apply, layer_apply, loss_apply, schedule="dual_pipe", grad_dtype=jnp.float32):
  """Build a serial or fused layer-level 1F1B loss/gradient computation.

  Callbacks:
    prefix_apply(boundary_params, data) -> hidden
    layer_apply(params, hidden, state, positions, segments) -> hidden
    loss_apply(boundary_params, hidden, data) -> (unnormalized_loss, aux)

  The returned step accepts (layer_params, layer_state, boundary_params, data).
  Layer parameters/state have a leading layer axis; every data leaf has a
  leading microbatch axis. The data mapping must contain inputs_position and
  inputs_segmentation. State is read-only and callbacks must be deterministic.

  Returns (loss_sum, aux_sum, layer_grads_sum, boundary_grads_sum). Boundary
  parameters are shared by prefix and head so tied embeddings receive BOTH
  contributions. All parameters stay fixed across the accumulation window.
  """
  if schedule not in ("serial", "dual_pipe"):
    raise ValueError(f"Unknown schedule: {schedule}")
  if not hasattr(jax, "fwd_and_bwd"):
    raise RuntimeError("The dual-pipe experiment requires JAX with jax.fwd_and_bwd")

  prefix_forward, prefix_backward = jax.fwd_and_bwd(prefix_apply, argnums=(0,), jitted=False)
  forward, backward = jax.fwd_and_bwd(layer_apply, argnums=(0, 1), jitted=False)
  # Reuse each forward trace's saved-VJP metadata between fill and steady state.
  # These helpers inline into the caller's train_step, not separate GPU calls.
  prefix_forward = jax.jit(prefix_forward, inline=True)
  prefix_backward = jax.jit(prefix_backward, inline=True)
  forward = jax.jit(forward, inline=True)
  backward = jax.jit(backward, inline=True)
  loss_and_grad = jax.jit(jax.value_and_grad(loss_apply, argnums=(0, 1), has_aux=True), inline=True)

  def step(layer_params, layer_state, boundary_params, microbatch_data):
    microbatches = microbatch_data["inputs_position"].shape[0]
    if microbatches < 1:
      raise ValueError("At least one microbatch is required")
    if any(x.shape[0] != microbatches for x in jax.tree.leaves(microbatch_data)):
      raise ValueError("All input leaves must have the same leading microbatch axis")

    def accumulate(total, grads):
      return jax.tree.map(lambda a, g: a + g.astype(grad_dtype), total, grads)

    def sum_aux(total, aux):
      return jax.tree.map(lambda a, b: a + b, total, aux)

    def forward_layers(hidden, data):
      def body(hidden, layer_data):
        weights, state = layer_data
        with jax.named_scope("forward"):
          return forward(weights, hidden, state, data["inputs_position"], data["inputs_segmentation"])

      return jax.lax.scan(body, hidden, (layer_params, layer_state))

    def backward_layers(residuals, dhidden):
      def body(dhidden, residual):
        with jax.named_scope("backward"):
          dweights, dhidden = backward(residual, dhidden)
        return dhidden, dweights

      # reverse=True visits L-1..0 but stacks gradients in original layer order.
      return jax.lax.scan(body, dhidden, residuals, reverse=True)

    def forward_microbatch(data):
      with jax.named_scope("prefix_forward"):
        hidden, prefix_residuals = prefix_forward(boundary_params, data)
      hidden, residuals = forward_layers(hidden, data)
      with jax.named_scope("head_loss_and_backward"):
        (loss, aux), (head_grads, dhidden) = loss_and_grad(boundary_params, hidden, data)
      return prefix_residuals, residuals, dhidden, loss, aux, head_grads

    first_data = jax.tree.map(lambda x: x[0], microbatch_data)
    later_data = jax.tree.map(lambda x: x[1:], microbatch_data)
    layer_total = jax.tree.map(lambda p: jnp.zeros(p.shape, grad_dtype), layer_params)
    boundary_total = jax.tree.map(lambda p: jnp.zeros(p.shape, grad_dtype), boundary_params)

    with jax.named_scope("fill_F0"):
      prefix_residuals, residuals, dhidden, loss_total, aux_total, head_grads = forward_microbatch(first_data)
    boundary_total = accumulate(boundary_total, head_grads)

    if schedule == "serial" or microbatches == 1:
      with jax.named_scope("serial_B0"):
        dprefix, layer_grads = backward_layers(residuals, dhidden)
        prefix_grads, = prefix_backward(prefix_residuals, dprefix)
      layer_total = accumulate(layer_total, layer_grads)
      boundary_total = accumulate(boundary_total, prefix_grads)

      def serial_microbatch(carry, data):
        loss_sum, aux_sum, layer_sum, boundary_sum = carry
        prefix_res, layer_res, dy, loss, aux, head_grads = forward_microbatch(data)
        dprefix, layer_grads = backward_layers(layer_res, dy)
        with jax.named_scope("prefix_backward"):
          prefix_grads, = prefix_backward(prefix_res, dprefix)
        boundary_sum = accumulate(accumulate(boundary_sum, head_grads), prefix_grads)
        return (loss_sum + loss, sum_aux(aux_sum, aux), accumulate(layer_sum, layer_grads), boundary_sum), None

      with jax.named_scope("serial_FB"):
        totals, _ = jax.lax.scan(
            serial_microbatch, (loss_total, aux_total, layer_total, boundary_total), later_data
        )
      return totals

    def backward_forward(carry, data):
      previous_prefix_residuals, previous_residuals, previous_dhidden, loss_sum, aux_sum, layer_sum, boundary_sum = carry
      with jax.named_scope("prefix_forward"):
        next_hidden, next_prefix_residuals = prefix_forward(boundary_params, data)
      reversed_residuals = jax.tree.map(lambda r: r[::-1], previous_residuals)

      def combined_layer(carry, layer_data):
        dhidden, next_hidden = carry
        old_residual, next_weights, next_state = layer_data
        # B_i at layer L-1-k and F_(i+1) at layer k have independent carries.
        with jax.named_scope("backward"):
          dweights, dhidden = backward(old_residual, dhidden)
        with jax.named_scope("forward"):
          next_hidden, next_residual = forward(
              next_weights, next_hidden, next_state, data["inputs_position"], data["inputs_segmentation"]
          )
        return (dhidden, next_hidden), (dweights, next_residual)

      with jax.named_scope("combined_bf_layers"):
        (dprefix, next_hidden), (reversed_grads, next_residuals) = jax.lax.scan(
            combined_layer, (previous_dhidden, next_hidden), (reversed_residuals, layer_params, layer_state)
        )
      layer_grads = jax.tree.map(lambda g: g[::-1], reversed_grads)
      with jax.named_scope("prefix_backward"):
        prefix_grads, = prefix_backward(previous_prefix_residuals, dprefix)
      with jax.named_scope("head_loss_and_backward"):
        (loss, aux), (head_grads, next_dhidden) = loss_and_grad(boundary_params, next_hidden, data)
      boundary_sum = accumulate(accumulate(boundary_sum, prefix_grads), head_grads)
      return (
          next_prefix_residuals,
          next_residuals,
          next_dhidden,
          loss_sum + loss,
          sum_aux(aux_sum, aux),
          accumulate(layer_sum, layer_grads),
          boundary_sum,
      ), None

    # F0, [B0 + F1], [B1 + F2], ..., B_last. Only the next microbatch's
    # residuals persist in the outer carry; old/new residuals can coexist inside.
    with jax.named_scope("steady_Bi_Fnext"):
      carry, _ = jax.lax.scan(
          backward_forward,
          (prefix_residuals, residuals, dhidden, loss_total, aux_total, layer_total, boundary_total),
          later_data,
      )
    prefix_residuals, residuals, dhidden, loss_total, aux_total, layer_total, boundary_total = carry
    with jax.named_scope("drain_Blast"):
      dprefix, layer_grads = backward_layers(residuals, dhidden)
      prefix_grads, = prefix_backward(prefix_residuals, dprefix)
    return loss_total, aux_total, accumulate(layer_total, layer_grads), accumulate(boundary_total, prefix_grads)

  return step
