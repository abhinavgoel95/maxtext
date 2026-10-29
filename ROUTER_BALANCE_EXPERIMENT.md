# Random-logit router benchmark

This experimental branch builds on MaxText PR #5487 (head
`a4c12a4fa12dfdf0d7f8b079559ba822b01c087a`). TE MoE runs require
TransformerEngine PR #3596, which provides `RoutingMapInfo` and routing-weight
gradients. Older TE installations cannot run this mode.

Add to an otherwise unchanged training command:

```bash
moe_router_force_load_balancing=true \
load_balance_loss_weight=0.0 \
routed_bias_update_rate=0.0
```

With maxtext-launcher, prefix each setting with `--maxtext-arg` and mount this
checkout using `--code-dir`.

The gate GEMM still runs. Its logits are replaced by standard-normal values
before activation and expert bias. A straight-through estimator passes router
gradients back to the original logits. Normal top-k selects distinct experts;
the existing `use_random_routing` option is not used. For TE, routing is computed
in MaxText and passed into the existing dispatch/expert/combine implementation.
This changes routing kernels/scheduling relative to the internal TE router.

This is synthetic routing for benchmarking, not equivalent model training.
Balance is statistical, not exact. Buffer sizing, overflow detection, and token
capacity policies are unchanged; do not disable overflow checks.

Initial restrictions: no auxiliary routing loss, adaptive bias updates, hash or
forced routing, serving quantization, or dropless fallback. Other settings remain
unchanged. Default-off runs retain their previous routing behavior.

Validation before large-scale use:

- Run `pytest tests/unit/moe_test.py -k ForceLoadBalancing` in a MaxText environment.
- Verify NNX RNG state advances across steps, microbatches and scanned layers.
  Pure JAX scan/remat tests alone do not establish this for the full model.
- Verify rematerialized routing replays exactly on multiple devices; distinct
  token shards must not repeat identical random patterns.
- Check finite loss/gradients, expert counts and zero capacity overflows in a
  small TE run before comparing performance. No GPU training run is implied by
  publication of this branch.
