# Dense dual-pipe scheduling in normal training

`gradient_accumulation_schedule=dual_pipe` opts the normal
`maxtext.trainers.pre_train.train` entry point into a layer-level fused 1F1B
gradient-accumulation schedule. The default is `serial`, which keeps the existing
training implementation. The launcher does not need a different Python module.

For three microbatches, the decoder schedule is `F0, B0F1, B1F2, B2`.
Each combined scan iteration contains backward for old layer `L-1-k` and forward
for next layer `k`, with independent activation carries. Both paths compile
inside the existing jitted `train_step`. This exposes overlap opportunities; it
does not guarantee GPU concurrency or implement distributed pipeline stages.

Unlike the standalone decoder microbenchmark, this path uses the actual token
embedding, final normalization, output head, and masked causal-LM loss. Output
cotangents come from that loss. Embedding/head and decoder gradients are all
accumulated, normalized by the same valid-token count as normal GA, and returned
in the original parameter tree/scan-axis layout. Existing clipping, optimizer,
checkpointing, evaluation, data loading, and profiling remain in normal training.
Parameters do not update between microbatches.

## Initial supported scope

- NNX dense Llama (`decoder_block=llama2`), including `model_name=llama3-8b`.
- `scan_layers=true`, homogeneous layers, and `shard_mode=auto`.
- `remat_policy=none` or `full`; `full` uses `nothing_saveable`.
- `dropout_rate=0`, no quantization, and float32/bfloat16/float16 parameters.
- `num_vocab_tiling=1`, ordinary causal-LM loss, no mutable internal metrics.
- No MoE, LoRA, multimodal, MTP/indexer, pipeline parallelism, host offload,
  Tunix accumulation, DiLoCo, or `shard_optimizer_over_data` (ZeRO-1).

Unsupported modes fail explicitly. Full remat recomputes the old microbatch's
forward operations during backward; those are distinct from the next forward
branch. Old and new residuals can coexist in the combined loop, so peak memory
is not guaranteed to be one microbatch's saved activations.

## One-node Llama 3 8B with maxtext-launcher on Lyris

From `/lustre/fsw/coreai_dlcompiler_ci/abgoel/jax_maxtext/maxtext-launcher`:

```bash
python3 launcher.py llama3-8b \
  --cluster lyris --nodes 1 --ici-fsdp 4 \
  --code-dir /lustre/fsw/coreai_dlcompiler_ci/abgoel/jax_maxtext/maxtext \
  --batch-size 1 --scan-layers --no-pgle --profiler xplane \
  --maxtext-arg gradient_accumulation_schedule=dual_pipe \
  --maxtext-arg gradient_accumulation_steps=3 \
  --maxtext-arg max_target_length=1024 \
  --maxtext-arg remat_policy=full \
  --maxtext-arg dropout_rate=0 \
  --maxtext-arg shard_mode=auto \
  --xla-flag xla_gpu_enable_command_buffer= \
  --tag llama8b-dualpipe-m3 \
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

With four GPUs, these batch settings give global microbatch size 4 and total
accumulated batch size 12. The model retains all 32 decoder layers; sequence
length 1024 is a small first-run configuration, not an 8K benchmark.
The launcher's Llama preset uses `minimal_with_context`, which must be overridden
to `full` or `none` for this path. Its default 21 steps cover the XPlane capture
window (skip 10, capture 3).

For comparison, change the schedule to `serial` and use a different tag. Keep
all other options identical. Normal launcher profiling/HLO paths apply to both
runs. Look for `dual_pipe/steady_Bi_Fnext/combined_bf_layers` containing both
`backward` and `forward` operations. Dense FSDP has all-gather/reduce-scatter
communication, not TE-MoE dispatch/combine A2As.

The container must support the mounted MaxText checkout and provide
`jax.fwd_and_bwd`. Initial validation is CPU-only: tiny true-loss gradient,
NNX bookkeeping, and structural scan tests. A full GPU Llama run and actual
communication overlap still require validation. The training path does not add
runtime numerical comparisons or NaN/Inf assertions.
