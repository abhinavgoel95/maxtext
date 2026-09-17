# Layer-fused dual-pipe scheduling in normal training

`gradient_accumulation_schedule=dual_pipe` opts the normal
`maxtext.trainers.pre_train.train` entry point into a layer-level fused 1F1B
gradient-accumulation schedule. The default is `serial`, which keeps the existing
training implementation. The launcher does not need a different Python module.

For three microbatches, the decoder schedule is `F0, B0F1, B1F2, B2`.
Each combined scan iteration contains backward for old layer `L-1-k` and forward
for next layer `k`, with independent activation carries. Both paths compile
inside the existing jitted `train_step`. This exposes overlap opportunities; it
does not guarantee GPU concurrency or implement distributed pipeline stages.

This path uses the actual token embedding, final normalization, output head,
and masked causal-LM loss. Output
cotangents come from that loss. Embedding/head and decoder gradients are all
accumulated, normalized by the same valid-token count as normal GA, and returned
in the original parameter tree/scan-axis layout. Existing clipping, optimizer,
checkpointing, evaluation, data loading, and profiling remain in normal training.
Parameters do not update between microbatches.

## Code layout

- `train.py` selects the schedule and keeps the normal optimizer/training loop.
  Its `loss_from_logits` helper shares the existing masked loss calculation;
  loss and z-loss normalization remain with the callers.
- `experimental/dense_training_schedule.py` implements the fused layer scans.
- `experimental/dense_training_nnx.py` adapts scanned Llama or DeepSeek layers and the
  embedding/head to those scans, then restores the full model gradient tree.

There is no separate benchmark entry point.

## Initial supported scope

- NNX dense Llama (`decoder_block=llama2`), including `model_name=llama3-8b`;
  or DeepSeek V3 (`decoder_block=deepseek`) with an initial dense stack and
  a TE-MoE stack, as described below.
- `scan_layers=true`, homogeneous layers within each stack, and `shard_mode=auto`.
- `remat_policy=none` or `full`; `full` uses `nothing_saveable`.
- `dropout_rate=0`, no quantized weights/activations. The current TE-MoE adapter
  requires `quantization=te_no_quant` and `te_gmm_quantization=te_no_quant`.
  The test scripts below explicitly set both modes to `te_no_quant`.
- DeepSeek router bias may be enabled with `routed_bias_update_rate=0` and
  `load_balance_loss_weight=0`; it remains read-only, non-parameter layer state.
- `num_vocab_tiling=1`, ordinary causal-LM loss, no mutable internal metrics.
- No non-TE MoE, LoRA, multimodal, MTP/indexer, pipeline parallelism, host offload,
  Tunix accumulation, DiLoCo, or `shard_optimizer_over_data` (ZeRO-1).

Unsupported modes fail explicitly. Full remat recomputes the old microbatch's
forward operations during backward; those are distinct from the next forward
branch. Old and new residuals can coexist in the combined loop, so peak memory
is not guaranteed to be one microbatch's saved activations.

## One-node Llama 3 8B with maxtext-launcher on Lyris

From `/lustre/fsw/coreai_dlcompiler_ci/abgoel/jax_maxtext/maxtext-launcher`:

```bash
python3 launcher.py llama3-8b \
  --cluster lyris --nodes 1 \
  --code-dir /lustre/fsw/coreai_dlcompiler_ci/abgoel/jax_maxtext/maxtext \
  --maxtext-arg gradient_accumulation_schedule=dual_pipe \
  --maxtext-arg gradient_accumulation_steps=3 \
  --maxtext-arg scan_layers=true \
  --maxtext-arg remat_policy=full \
  --tag llama8b-dualpipe-ga3-full-remat \
  --dry-run
```

Inspect the generated scripts, then rerun without `--dry-run` to submit. Leave
their Python entry point as `maxtext.trainers.pre_train.train`. No generated
script editing is needed. `code_dir` mounts the host working tree at
`/opt/maxtext`; alternatively put the same path in the launcher's personal YAML:

```yaml
code_dir: "/lustre/fsw/coreai_dlcompiler_ci/abgoel/jax_maxtext/maxtext"
code_mount: "/opt/maxtext"
maxtext_path: "/opt/maxtext"
```

The command leaves batch size, sequence length, PGLE, command buffers, and
profiling at the launcher's configured values. The tested Llama preset uses
four GPUs, 32 layers, sequence length 8192, and per-device batch size 2:
global microbatch size 8 and accumulated batch size 24 with GA=3.
The launcher's Llama preset uses `minimal_with_context`, which must be overridden
to `full` or `none` for this path. Its default 21 steps cover the XPlane capture
window (skip 10, capture 3).

For comparison, change the schedule to `serial` and use a different tag. Keep
all other options identical. Normal launcher profiling/HLO paths apply to both
runs. Look for `dual_pipe/steady_Bi_Fnext/combined_bf_layers` containing both
`backward` and `forward` operations. Dense FSDP has all-gather/reduce-scatter
communication, not TE-MoE dispatch/combine A2As.

The container must support the mounted MaxText checkout and provide
`jax.fwd_and_bwd`. Focused CPU tests cover true-loss gradients, NNX bookkeeping,
loss masking, and scan structure. The pre-cleanup implementation completed a
21-step four-GPU Llama 3 8B run; its optimized HLO showed both B/F interleaving
and asynchronous collective windows. That is not a numerical equivalence test
or proof of GPU overlap. The training path does not add runtime numerical
comparisons or NaN/Inf assertions.

## One-node reduced-layer DeepSeek V3 with TE-MoE

Two matched scripts inherit the regular `deepseek-v3-671b` launcher preset and
use the normal training entry point, without creating another model config.
The reference is `ds-v3-JAX_ENABLE_X64-0-correctefd_20260917_092949`.

```bash
cd /lustre/fsw/coreai_dlcompiler_ci/abgoel/jax_maxtext/maxtext
bash scripts/experimental/run_deepseek_small_serial.sh --dry-run
bash scripts/experimental/run_deepseek_small_dualpipe.sh --dry-run
```

Inspect the generated scripts first. Remove `--dry-run` to submit a test.
Both test configurations pass the adapter's feature checks. The dual-pipe
configuration completed a 15-step GPU smoke test; numerical equivalence and
performance against serial remain to be established.
Other launcher options can be appended unchanged, for example
`--profiler nsys` or `--container IMAGE`. Both scripts now pass `--no-pgle`
for the runtime-crash investigation; XPlane remains the default profiler.
The only environment override provided
by these scripts is `LAUNCHER_DIR`, if the launcher is installed elsewhere.
The MaxText code mount is derived from the script's own checkout.

The commands differ only in the schedule and tag. Their only training overrides
relative to the regular preset are:

- One node, four one-GPU processes, FSDP=2, EP=2; every DCN axis is 1.
- Six layers: `base_num_decoder_layers=6`, with `first_num_dense_layers=1`.
- Sixteen routed experts: `num_experts=16`.
- GA=3 and the selected accumulation schedule.
- Full rematerialization (`remat_policy=full`) instead of the preset's custom policy.
- No quantization: `quantization=te_no_quant` and `te_gmm_quantization=te_no_quant`
  explicitly override the preset's FP8 modes in both tests.
- `override_model_config=true` so the three model-size overrides take effect.
- PGLE disabled (`--no-pgle`) to isolate the serial-run illegal memory access.

Everything else comes from the regular preset and DeepSeek model configuration:
top-8 routing, embedding width 7168, dense/expert MLP widths 18432/2048,
128 query/KV heads, Q/KV LoRA ranks 1536/512, sequence length 4096,
per-device batch size 6, BF16 weights and `mu_dtype`,
router bias, `capacity_factor=1.0`,
`ragged_buffer_factor=2.0`, and 15 steps. XPlane, command buffers,
NCCL settings, and `JAX_ENABLE_X64=0` are also inherited unchanged.
The global microbatch is 24 and the accumulated batch is 72 with GA=3.
The preset's per-activation remat/offload settings are not edited, but selecting
`remat_policy=full` replaces its custom remat/offload policy.

The launcher currently prints the preset's GA=1 in its summary even though the
emitted Python command correctly contains `gradient_accumulation_steps=3`.
Cluster queue/account/time settings remain those of the current launcher config;
the saved reference used `gb300`, while current Lyris defaults may select `gb200`.

The combined decoder body pairs old/new logical layers as follows:

```text
old B(layer 5, MoE)   + next F(layer 0, dense)
old B(layer 4, MoE)   + next F(layer 1, MoE)
old B(layer 3, MoE)   + next F(layer 2, MoE)
old B(layer 2, MoE)   + next F(layer 3, MoE)
old B(layer 1, MoE)   + next F(layer 4, MoE)
old B(layer 0, dense) + next F(layer 5, MoE)
```

The mixed dense/MoE boundaries need distinct scan segments because their
parameter and saved-residual trees differ. Within each segment, backward and
next forward still share the same compiled loop body. Rematerialized old-forward
work inside backward is distinct from the next-microbatch forward branch.

The adapter allows the inherited `routed_bias=true` with its update rate at zero.
The existing per-layer state path carries the bias without parameter gradients
or optimizer updates. Nonzero bias updates, auxiliary load-balancing loss, and
FP8 remain unsupported. No routing settings or kernels were changed to allow
frozen bias. A one-node GPU run on 2026-09-17 completed all 15 steps with these
dual-pipe settings. Its optimized HLO contains next-forward EP combine windows
spanning backward grouped GEMMs, and backward EP combine windows spanning
next-forward dense GEMMs. This establishes execution and scheduling structure,
not numerical equivalence to serial or measured GPU overlap. Some dispatch
start/done pairs remain adjacent.

Use a compatible container and four one-GPU processes; normal MaxText setup
performs EP bootstrap. CPU tests cannot validate TE communication,
routing-handle lifetime under scan/remat,
or pinned-host residual handling in the combined scans.
