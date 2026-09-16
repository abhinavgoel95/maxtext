# Dense decoder fused-BF scheduling experiment

For the normal MaxText training entry point and launcher, see
[Dense dual-pipe training](dense_dualpipe_training.md). The standalone harness
below remains an optional decoder-only profiling tool.

This is a standalone scheduling/profile harness, not a replacement for MaxText
training. It uses actual MaxText Llama decoder attention, normalization, MLP and
residual operations, with synthetic hidden states and output cotangents.
Embeddings, vocabulary projection, cross-entropy and optimizer are excluded.

Both schedules use the same initialized weights and seeded inputs. Each returns
outputs, summed parameter gradients, and input gradients. There are no model
numerical-comparison or NaN/Inf assertions. Completion does not certify numerical
correctness. Structural unit tests cover shapes and the shared BF scan body.

## Schedule to review

For three microbatches, `serial` is `F0,B0,F1,B1,F2,B2` and `dual_pipe` is
`F0,B0F1,B1F2,B2`. At combined layer-loop iteration k:

- Backward uses microbatch i, layer L-1-k and its saved forward residual.
- Forward uses microbatch i+1, layer k and the same fixed parameter version.
- The backward activation and forward activation are separate loop carries.

Both paths live in the same inner scan body and the whole step is one executable.
The inline JIT helpers stabilize first-class VJP metadata; Python does not launch
them separately. This exposes independent work to XLA; it does not require the
GPU to overlap it. This is dual-pipe-like local scheduling, not full distributed
DualPipe with bidirectional pipeline stages.

The outer carry retains the next microbatch's residuals and one accumulated dW
tree. Gradients remain layer-leading, rather than being restored to TrainState.
There is no optimizer update or training-loss normalization in this experiment.

## Initial supported configuration

Use a homogeneous `decoder_block=llama2` stack, `scan_layers=true`,
`shard_mode=auto`, `inhomogeneous_layer_cycle_interval=1`, `dropout_rate=0`, no
quantization, no MoE, no pipeline parallelism, and no mutable internal metrics.
Unsupported stateful modes fail explicitly. Start with `remat_policy=none`;
`remat_policy=full` uses `nothing_saveable` identically in both schedules.

Model dimensions, dtype, batch, mesh, attention backend and profiling controls
come from normal MaxText configuration. `gradient_accumulation_steps` is the
microbatch count. Inputs are generated directly in microbatch layout, so this
does not reinterpret the ordering of any real training batch.

## Proposed launch pattern (after installing the diff)

Supply a compatible dense model config and the normal MaxText container/launcher.
The exact GPU/attention/mesh configuration must still be selected and tested.
All experiment flags precede the config filename; MaxText overrides follow it.

```bash
python -m maxtext.experimental.run_dense_schedule \
  --schedule serial --output-dir ./dense_schedule_runs \
  /path/to/dense.yml \
  run_name=dense_serial gradient_accumulation_steps=3 \
  remat_policy=none dropout_rate=0 shard_mode=auto steps=10

python -m maxtext.experimental.run_dense_schedule \
  --schedule dual_pipe --output-dir ./dense_schedule_runs \
  /path/to/dense.yml \
  run_name=dense_dualpipe gradient_accumulation_steps=3 \
  remat_policy=none dropout_rate=0 shard_mode=auto steps=10
```

For XPlane, add `profiler=xplane` and the desired capture window using
`skip_first_n_steps_for_profiler` and `profiler_steps`. Use enough measured steps
to cover that window. Traces use MaxText's configured `tensorboard_dir`, derived
from `base_output_directory` and `run_name`, not the local `--output-dir`.
For Nsight, launch with `nsys profile --capture-range=cudaProfilerApi` and use
`profiler=nsys`; the existing MaxText Profiler supplies CUDA start/stop calls.
Use per-rank Nsight report names in a multiprocess launcher.

Initialization, compilation, and one warmup run happen outside capture. Each
measured iteration synchronizes its result. Reported wall time includes profiler
boundary/flush costs when profiling is enabled; use unprofiled runs for timings.

Optimized HLO is always written to
`<output-dir>/<run_name>/<schedule>/rank_<rank>_after_optimizations.hlo.txt`.
No XLA dump-name filter is needed. Look for `fill_F0`, `steady_Bi_Fnext`,
`combined_bf_layers/{backward,forward}` and `drain_Blast`. XLA may transform loops,
so inspect the optimized result rather than assuming every source scan survives.

Compare both modes with identical compiler flags, remat and shardings. Dense
layers have no MoE dispatch/combine A2A. The mesh/attention configuration determines
which collectives, if any, are available to overlap. Verify actual concurrency
and critical-path changes in Nsight/XPlane, and record peak memory too.

## Validation status of the review draft

Based on MaxText main `9624c747973127e96cd2971c1289b3b468bb34b0`.
Core CPU tests pass with JAX 0.11.1 for M=1,2,3; a JAXPR check finds forward and
backward in one L=4 steady-state scan body. This is not a GPU overlap result.
The MaxText/NNX adapter and runner have been source-reviewed and syntax-checked,
but not executed with the full MaxText dependencies or the target container.
