#!/usr/bin/env bash
# Run this matched small DeepSeek TE-MoE case through the normal MaxText launcher.
# Add --dry-run to inspect generated files without submitting a SLURM job.
set -euo pipefail

LAUNCHER_DIR="${LAUNCHER_DIR:-/lustre/fsw/coreai_dlcompiler_ci/abgoel/jax_maxtext/maxtext-launcher}"
MAXTEXT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

exec python3 "${LAUNCHER_DIR}/launcher.py" deepseek-v3-671b \
  --cluster lyris --nodes 1 --ntasks-per-node 4 --sbatch-segment 0 \
  --code-dir "${MAXTEXT_DIR}" \
  --batch-size 1 --scan-layers --profiler xplane \
  --ici-dp 1 --ici-fsdp 2 --ici-tp 1 --ici-expert 2 \
  --dcn-dp 1 --dcn-fsdp 1 --dcn-tp 1 --dcn-expert 1 \
  --quantization te_no_quant \
  --maxtext-arg override_model_config=true \
  --maxtext-arg gradient_accumulation_schedule=dual_pipe \
  --maxtext-arg gradient_accumulation_steps=3 \
  --maxtext-arg base_num_decoder_layers=4 \
  --maxtext-arg first_num_dense_layers=1 \
  --maxtext-arg base_emb_dim=2048 \
  --maxtext-arg base_num_query_heads=16 \
  --maxtext-arg base_num_kv_heads=16 \
  --maxtext-arg base_mlp_dim=4096 \
  --maxtext-arg base_moe_mlp_dim=1024 \
  --maxtext-arg q_lora_rank=512 \
  --maxtext-arg kv_lora_rank=128 \
  --maxtext-arg num_experts=16 \
  --maxtext-arg num_experts_per_tok=2 \
  --maxtext-arg max_target_length=1024 \
  --maxtext-arg dtype=bfloat16 \
  --maxtext-arg weight_dtype=float32 \
  --maxtext-arg mu_dtype=float32 \
  --maxtext-arg remat_policy=full \
  --maxtext-arg te_moe_block=true \
  --maxtext-arg te_gmm_quantization=te_no_quant \
  --maxtext-arg sparse_matmul=true \
  --maxtext-arg prefuse_moe_weights=true \
  --maxtext-arg load_balance_loss_weight=0.0 \
  --maxtext-arg routed_bias=false \
  --maxtext-arg routed_bias_update_rate=0.0 \
  --maxtext-arg norm_topk_prob=false \
  --maxtext-arg use_random_routing=false \
  --maxtext-arg ragged_buffer_factor=-1.0 \
  --maxtext-arg capacity_factor=-1.0 \
  --maxtext-arg steps=21 \
  --maxtext-arg skip_first_n_steps_for_profiler=4 \
  --maxtext-arg profiler_steps=3 \
  --tag deepseek-small-dualpipe-ga3 \
  "$@"
