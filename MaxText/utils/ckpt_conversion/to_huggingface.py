# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import jax
import os
from typing import Sequence, Dict, Any
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer
from absl import app
import flax

from MaxText import max_utils
from MaxText import maxengine
from MaxText import pyconfig
from MaxText import max_logging

from MaxText.utils.ckpt_conversion.utils.param_mapping import (
    HOOK_FNS,
    PARAM_MAPPING,
)
from MaxText.utils.ckpt_conversion.utils.shape_mapping import SHAPE_MAPPING
from MaxText.utils.ckpt_conversion.utils.hf_model_configs import HF_MODEL_CONFIGS
from MaxText.utils.ckpt_conversion.utils.utils import (process_leaf_param, save_model_files, HF_IDS)

"""Convert MaxText unscanned ckpt into HF format"""


def _get_model_mappings(model_name: str, scan_layers: bool, config_dict: dict):  # Changed config to config_dict
  """Retrieves parameter, shape, and hook function mappings for the model."""
  if model_name not in PARAM_MAPPING or model_name not in SHAPE_MAPPING or model_name not in HOOK_FNS:
    raise ValueError(f"Mappings not found for model: {model_name}. Available PARAM_MAPPING keys: {PARAM_MAPPING.keys()}")

  return {
      "param_mapping": PARAM_MAPPING[model_name](config_dict, scan_layers),
      "shape_mapping": SHAPE_MAPPING[model_name](config_dict),
      "hook_fn_mapping": HOOK_FNS[model_name](config_dict, scan_layers, saving_to_hf=True),
  }


# Read Hugging Face token from environment variable
hf_token = os.environ.get("HF_AUTH_TOKEN")


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  config = pyconfig.initialize(argv)
  assert (
      config.load_full_state_path == ""
  ), "This script expects parameters, not a full state. Use generate_param_only_checkpoint first if needed."
  max_utils.print_system_information()

  engine = maxengine.MaxEngine(config)
  rng = jax.random.PRNGKey(1234)
  rng, rng_load_params = jax.random.split(rng)
  # load params from maxengine
  loaded_params_from_engine = engine.load_params(rng_load_params)

  if not config.base_output_directory:
    output_directory = os.path.expanduser("~/.hf_output/")
  else:
    output_directory = config.base_output_directory

  # 1. Get HuggingFace Model Configuration
  model_key = config.model_name
  if model_key not in HF_MODEL_CONFIGS:
    raise ValueError(f"HF configuration not found for model key: {model_key}")
  hf_config_obj = HF_MODEL_CONFIGS[model_key]

  # 2. Load Tokenizer
  if model_key not in HF_IDS:
    raise ValueError(f"HF Tokenizer ID not found for model key: {model_key}")
  hf_tokenizer_id = HF_IDS[model_key]
  tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_id, token=hf_token)

  # 3. Get parameter mappings
  mappings = _get_model_mappings(model_key, config.scan_layers, hf_config_obj.to_dict())
  param_map = mappings["param_mapping"]
  shape_map = mappings["shape_mapping"]  # HF target shapes
  hook_fn_map = mappings["hook_fn_mapping"]

  # 4. Transform Weights
  transformed_hf_weights: Dict[str, Any] = {}

  # MaxText `engine.load_params()` returns `state.params` (a FrozenDict).
  # The actual weights are typically under `state.params['params']`.
  actual_weights_dict = loaded_params_from_engine.get("params")
  if actual_weights_dict is None:
    raise ValueError("Loaded parameters from engine do not contain a 'params' key. Structure might be unexpected.")

  leaves_with_paths = jax.tree_util.tree_leaves_with_path(actual_weights_dict)

  # traverse leavse to build: mt_param_key:mt_weights
  processed_params_list = []
  for path_tuple_iter, leaf_value_iter in leaves_with_paths:
    processed_params_list.extend(
        process_leaf_param(path_tuple_iter, leaf_value_iter, param_map, shape_map, hook_fn_map, config)
    )
  transformed_hf_weights = dict(processed_params_list)

  # 5. Save in HuggingFace Format
  if not transformed_hf_weights:
    print("Error: No weights were transformed. Check mappings and parameter paths.")
    return

  save_model_files(
      weight_arrays=transformed_hf_weights,
      config=hf_config_obj,
      tokenizer=tokenizer,
      output_dir=output_directory,
  )
  max_logging.log(f"✅ MaxText model successfully saved in HuggingFace format at {output_directory}")


if __name__ == "__main__":
  app.run(main)
