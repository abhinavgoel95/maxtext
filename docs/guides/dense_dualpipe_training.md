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
- `dropout_rate=0`, no quantized weights/activations. The TE-MoE case selects
  `quantization=te_no_quant` and `te_gmm_quantization=te_no_quant` explicitly.
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

## One-node small DeepSeek V3 with TE-MoE

Two matched scripts use the existing `deepseek-v3-671b` launcher preset and the
normal training entry point, without creating another model config:

```bash
cd /lustre/fsw/coreai_dlcompiler_ci/abgoel/jax_maxtext/maxtext
bash scripts/experimental/run_deepseek_small_serial.sh --dry-run
bash scripts/experimental/run_deepseek_small_dualpipe.sh --dry-run
```

Inspect the generated scripts first. Remove `--dry-run` to submit, starting with
serial. Other launcher options can be appended unchanged, for example
`--profiler nsys --no-pgle` or `--container IMAGE`. Disable PGLE when using nsys
instead of the preset's XPlane profiler. The only environment override provided
by these scripts is `LAUNCHER_DIR`, if the launcher is installed elsewhere.
The MaxText code mount is derived from the script's own checkout.

The commands differ only in the schedule and tag. Both set:

- One node, four one-GPU processes, FSDP=2, EP=2; every DCN axis is 1.
- Four layers: one dense MLA/MLP layer, followed by three MLA/TE-MoE layers.
- Sixteen routed experts, top-2 routing, one shared expert per MoE layer.
- Embedding width 2048, 16 query/KV heads, dense MLP width 4096, expert MLP
  width 1024, Q/KV LoRA ranks 512/128. QK-nope/QK-rope/V dimensions remain
  128/64/128, and the DeepSeek vocabulary remains 129280.
- Sequence length 1024, per-device batch 1, GA=3: global microbatch 4,
  accumulated global batch 12. Expert count and model widths are divisible by
  the selected two-way EP/FSDP axes.
- BF16 computation, FP32 weights/gradients/optimizer first moments, full remat,
  scanned layers, and `override_model_config=true` so shape/routing overrides
  actually replace the upstream DeepSeek model defaults.
- Twenty-one steps with XPlane capture after four steps for three steps.
  PGLE, command buffers, and the remaining XLA/environment settings are inherited
  from the launcher preset; neither script silently changes them.

This is a small **DeepSeek-shaped scheduling test**, not the 671B model or a
quality-training recipe. Both schedules explicitly set
`load_balance_loss_weight=0`, `routed_bias=false`, and
`routed_bias_update_rate=0`: the fused path does not implement an auxiliary
load-balancing objective or mutable router-bias updates. `ragged_buffer_factor=-1`
requests worst-case TE receive capacity, avoiding capacity-driven token dropping;
`capacity_factor=-1` is also explicit. TE receive/overflow metrics are retained,
not replaced with assumed success. Dropout, QK clipping, internal activation
metrics, indexer/MTP, batch-split scheduling, and quantization state updates are
disabled by the shared model/base defaults; the scripts do not repeat those
unchanged settings. The remaining ICI/DCN axes also retain their base value 1.

The combined decoder body pairs old/new logical layers as follows:

```text
old B(layer 3, MoE)   + next F(layer 0, dense)
old B(layer 2, MoE)   + next F(layer 1, MoE)
old B(layer 1, MoE)   + next F(layer 2, MoE)
old B(layer 0, dense) + next F(layer 3, MoE)
```

The mixed dense/MoE boundaries need distinct scan segments because their
parameter and saved-residual trees differ. Within each segment, backward and
next forward still share the same compiled loop body. Full remat additionally
recomputes old-microbatch forward work inside backward, including TE routing
communication where needed; those calls are not the next-microbatch forward.

This DeepSeek TE-MoE path is **not yet GPU-validated**. CPU schedule tests cannot
validate NCCL EP communication or TE's routing-handle lifetime under scan/remat.
Use a compatible container with TE MoE/EP support and four one-GPU processes;
normal MaxText setup performs EP bootstrap before model execution. A successful
serial run is the first runtime check before comparing dual-pipe HLO/profiles.
Passing execution does not establish numerical equivalence or GPU overlap.
