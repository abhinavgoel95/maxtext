# DeepSeek dual-pipe memory investigation

## Findings

Initial XLA allocation, GiB per GPU, on 16 nodes × 4 GB300 GPUs,
31 decoder layers (3 dense + 28 MoE), 256 experts:

| Schedule | GA | Temporary arena | Other allocations | Total |
| --- | ---: | ---: | ---: | ---: |
| Standard serial | 3 | 128.31 | 28.48 | 156.79 |
| Dual-pipe | 4 | 218.87 | 28.48 | 247.35 |

The 90.56 GiB difference is almost entirely temporary allocation. **GA differs:
this is an observation, not a controlled measurement of schedule-only overhead.**
Neither run reached a training step or PGLE recompilation: both encountered
a Triton scalar-collective LLVM struct-packing error after emitting HLO and
buffer assignments. These are compiler allocations, not measured peak HBM use.

Both MLA latent constraints are included. The earlier full-batch projection
intermediates are absent: local query output is `bf16[6,4096,128,192]`
(1.125 GiB), and local KV output is `bf16[6,4096,128,256]` (1.5 GiB).

The mixed dense/MoE schedule splits the combined layer loop into 3/25/3
iterations. Each segment returns saved forward residuals; the 25-layer and
3-layer MoE pieces are concatenated into the next microbatch's 28-layer stack.
The homogeneous-group shared-storage path does not apply to this mixed model.

Buffer assignment contains:
- `bf16[28,6,4096,128,128]`: 21 GiB.
- `bf16[25,6,4096,128,128]`: 18.75 GiB; HLO shows a segment
  dynamic-update-slice and a double-buffer clone.
- `bf16[28,6,4096,7168]`: 9.1875 GiB.
- `bf16[25,6,4096,7168]`: 8.203125 GiB.
- FP32 expert-weight-shaped buffers `f32[28,32,896,4096]`: 12.25 GiB.

These are individual tensor sizes, **not additive peak-memory contributions**.
The exact named checkpoints and simultaneous peak lifetimes still need tracing.
We have not established that all excess storage is necessary, or that it is an
XLA defect rather than storage requested by the adapter.

## 1. Get the experiment source and container

Use `ghcr.io/nvidia/jax:maxtext-2026-10-14`. The experiment source revision is
`2ffd86c54`; the branch also contains this documentation.

```bash
git clone --single-branch --branch abgoel/dualpipe-mla-latent-sharding \
  https://github.com/abhinavgoel95/maxtext.git maxtext-memory-repro
cd maxtext-memory-repro
git checkout 2ffd86c54
```

Mount this checkout inside the container (for example at `/opt/maxtext`).
This revision includes custom-remat support, the TE-EP per-call capacity fix,
and the query/KV latent constraints. It is an experimental reproduction branch,
not a claim that these changes are ready to merge.

## 2. Allocate and launch

Allocate 16 nodes × 4 GPUs on a single ICI fabric. Use one process per GPU,
64 processes total, through your usual MaxText distributed launcher. Use the
same host-memory and GPU resources in both runs. All DCN axes are 1.
Allocation/container startup is site-specific; no private filesystem paths or
launcher presets are required.

Execute the following **inside each container rank**, from the checkout.
Set `RUN_NAME` identically across ranks and uniquely for each job, and use a
shared writable `OUTPUT_ROOT`. The Slurm rank environment must be propagated
for the normal MaxText GPU multiprocess bootstrap. This block is a per-rank
command, not a command to run once on the login node.

To reproduce the observed table, use serial/GA3 and dual_pipe/GA4.
For a controlled comparison, use **GA4 in both** (serial GA4 is not yet measured).

```bash
cd /opt/maxtext
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
SCHEDULE=dual_pipe   # serial for ordinary MaxText GA; not adapter_serial
GA=4                # use 3 only to reproduce the historical serial row
RUN_NAME="memory-${SCHEDULE}-ga${GA}-trial1"
OUTPUT_ROOT=/output # mount a shared writable output directory here
mkdir -p "${OUTPUT_ROOT}/${RUN_NAME}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="0.88"
export CUDA_DEVICE_MAX_CONNECTIONS="16"
export NVTE_FUSED_ATTN="1"
export DECOUPLE_GCLOUD="TRUE"
export XLA_PJRT_GPU_HOST_MEMORY_PREALLOCATE="false"
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB="180"
export NVTE_MAXTEXT_DO_FSDP_AG_IN_BF16="1"
export TF_CPP_VMODULE="compilation_stats=10"
export TF_CPP_MIN_LOG_LEVEL="0"
export TF_CPP_MAX_LOG_LEVEL="10"
export JAX_ENABLE_PGLE=true
export JAX_PGLE_PROFILING_RUNS=3
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_all_reduce_combine_threshold_bytes=33554432 --xla_gpu_all_gather_combine_threshold_bytes=6442450944 --xla_gpu_reduce_scatter_combine_threshold_bytes=201326592 --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_disable_hlo_passes=rematerialization --xla_gpu_enable_command_buffer='FUSION,CUBLAS,CUDNN,DYNAMIC_SLICE_FUSION' --xla_gpu_experimental_enable_nccl_symmetric_buffers=false --xla_gpu_experimental_use_ragged_dot_fusion=true --xla_gpu_enable_cublaslt=false --xla_gpu_experimental_max_unroll_factor=8 --xla_gpu_print_compilation_stats=true --xla_gpu_memory_limit_slop_factor=99"
export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_as_text=true --xla_dump_to=${OUTPUT_ROOT}/${RUN_NAME}/hlo"

python -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  run_name="${RUN_NAME}" \
  base_output_directory="${OUTPUT_ROOT}/${RUN_NAME}" \
  jax_cache_dir="${OUTPUT_ROOT}/${RUN_NAME}/jax_cache" \
  model_name=deepseek3-671b \
  max_target_length=4096 \
  per_device_batch_size=6 \
  gradient_accumulation_steps="${GA}" \
  steps=15 \
  attention=cudnn_flash_jax \
  remat_policy=custom \
  enable_checkpointing=false \
  dataset_type=synthetic \
  dataset_path=local \
  hardware=gpu_multiprocess \
  quantization=te_fp8_currentscaling \
  profiler=xplane \
  skip_first_n_steps_for_profiler=4 \
  profiler_steps=3 \
  scan_layers=true \
  shardy=true \
  sparse_matmul=true \
  ici_data_parallelism=1 \
  ici_fsdp_parallelism=8 \
  ici_tensor_parallelism=1 \
  ici_expert_parallelism=8 \
  dcn_data_parallelism=1 \
  dcn_fsdp_parallelism=1 \
  dcn_tensor_parallelism=1 \
  dcn_expert_parallelism=1 \
  shard_optimizer_over_data=false \
  weight_dtype=bfloat16 \
  mu_dtype=bfloat16 \
  logits_dot_in_fp32=false \
  use_iota_embed=false \
  enable_goodput_recording=false \
  monitor_goodput=false \
  n_routing_groups=-1 \
  topk_routing_group=-1 \
  capacity_factor=1.0 \
  megablox=false \
  te_moe_block=true \
  te_gmm_quantization=te_mxfp8 \
  ragged_buffer_factor=1.0 \
  te_ep_overflow_check_every_n_steps=20 \
  prefuse_moe_weights=true \
  shard_exp_on_fsdp=false \
  mlpwi=device \
  mlpwi_0=device \
  mlpwi_1=device \
  mlpwo=device \
  moe_mlpwi_0=offload \
  moe_mlpwi_1=offload \
  moe_mlpwo=device \
  query_proj=remat \
  key_proj=remat \
  value_proj=remat \
  query_wa_proj=device \
  kv_wa_proj=device \
  out_proj=device \
  context=device \
  override_model_config=true \
  base_num_decoder_layers=31 \
  first_num_dense_layers=3 \
  num_experts=256 \
  gradient_accumulation_schedule="${SCHEDULE}" \
  load_balance_loss_weight=0.0 \
  routed_bias_update_rate=0.0
```

The custom policy saves query/KV low-rank projections, context and output
projection on device; rematerializes query/key/value projections; and offloads
MoE WI0/WI1. Do not substitute full remat when comparing these measurements.

The bootstrap should report `max_tokens_per_rank=24576` and
`recv_capacity_per_rank=196608` for both GA values. Buffer factor 1.0 is retained
to match the experiments. An earlier 8-node execution hit receive-capacity
overflow; successful execution/numerical validity is not implied by obtaining
compiler memory reports. Treat any capacity adjustment as a separate experiment.

## 3. Compare the compiler artifacts

Collect the initial `jit_train_step` optimized HLO,
`*-buffer-assignment.txt`, and `*-memory-usage-report.txt`.
Module numbers vary by rank/build. Do not confuse initial compilation with
later PGLE recompilations, and do not sum reports from different ranks.

```bash
find "${OUTPUT_ROOT}/${RUN_NAME}/hlo" -name '*jit_train_step*memory-usage-report.txt'
# For one INITIAL report / corresponding assignment:
grep 'Total bytes:' /path/to/initial-memory-usage-report.txt
grep 'preallocated-temp' /path/to/initial-buffer-assignment.txt
```

Inspect `combined_bf_layers`, the 25-layer residual outputs,
`dynamic_update_slice`, `concatenate`, and `double_buffer_clone`.
Trace producer/consumer lifetimes and allocation offsets before attributing
bytes to individual causes; aliases and reused storage must not be counted twice.
Check that neither MLA projection has a full-batch expanded output.

This is a workload-level reproduction recipe. A smaller standalone JAX
reproducer and a matched-GA attribution of the memory difference remain open.

