#!/usr/bin/env bash
# Keep the regular DeepSeek preset with test size, topology, GA, schedule, and full remat.
# Explicitly keep both TE paths unquantized; do not inherit the preset's FP8 modes.
# Disable PGLE for the current runtime-crash investigation.
# Add --dry-run to inspect generated files without submitting a SLURM job.
set -euo pipefail

LAUNCHER_DIR="${LAUNCHER_DIR:-/lustre/fsw/coreai_dlcompiler_ci/abgoel/jax_maxtext/maxtext-launcher}"
MAXTEXT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

exec python3 "${LAUNCHER_DIR}/launcher.py" deepseek-v3-671b \
  --cluster lyris --nodes 1 --ntasks-per-node 4 --sbatch-segment 0 \
  --code-dir "${MAXTEXT_DIR}" \
  --no-pgle \
  --ici-dp 1 --ici-fsdp 2 --ici-tp 1 --ici-expert 2 \
  --dcn-dp 1 --dcn-fsdp 1 --dcn-tp 1 --dcn-expert 1 \
  --maxtext-arg override_model_config=true \
  --maxtext-arg gradient_accumulation_schedule=serial \
  --maxtext-arg gradient_accumulation_steps=3 \
  --maxtext-arg remat_policy=full \
  --maxtext-arg quantization=te_no_quant \
  --maxtext-arg te_gmm_quantization=te_no_quant \
  --maxtext-arg base_num_decoder_layers=6 \
  --maxtext-arg first_num_dense_layers=1 \
  --maxtext-arg num_experts=16 \
  --tag deepseek-small-serial-ga3 \
  "$@"
